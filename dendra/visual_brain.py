"""
DENDRA-Brain for vision (Atari): visual cortex + dendritic conjunctions + striatum + hippocampus + world model.
Every part learns online with local rules; there is no backpropagation anywhere.

    frame ─► visual cortex (V1, V2) ─► sparse code ─┬─► dendritic conjunctions (pairs of moving features,
                                                    │    coded by their relative position)
                                                    ├─► striatum: Q(s, a) = Σ w over active features
                                                    │      learns from dopamine δ × eligibility tags (Sarsa(λ))
                                                    ├─► hippocampus: episodic memory of returns (long-term memory)
                                                    │      recall by similarity (k nearest neighbours)
                                                    └─► cerebellum: forward model of the next code and reward
                                                           planning = mental simulation that trains the striatum
    action = softmax over [striatum + hippocampus] values (arbitration)
"""
import math

import numpy as np
import torch

from .brain import Dopamine
from .vision import VisualCortex


class Habituation:
    """Sensory adaptation: a feature that is almost always on (walls, background) stops driving conjunctions."""

    def __init__(self, n_features, rate=0.01, limit=0.5):
        self.mean = torch.zeros(n_features)
        self.rate, self.limit = rate, limit

    def __call__(self, codes):
        fired = torch.zeros_like(self.mean)
        for c in codes:
            fired[c] += 1.0 / len(codes)
        self.mean += self.rate * (fired - self.mean)
        return [c[self.mean[c] < self.limit] for c in codes]


class DendriticConjunctions:
    """Pairs of non-habituated V1 complex features, coded by (type 1, type 2, relative offset): a dendritic
    branch that fires when two inputs coincide at a given spatial relation (e.g. "ball 2 cells left of paddle")."""

    def __init__(self, n_v1, grid=10, max_features=24):
        self.n_v1, self.grid, self.max = n_v1, grid, max_features
        self.span = 2 * grid - 1
        self.size = n_v1 * n_v1 * self.span * self.span

    def __call__(self, v1_indices):
        out = []
        for idx in v1_indices:
            idx = idx[: self.max]
            if len(idx) < 2:
                out.append(idx.new_zeros(0)); continue
            cell, pos = idx % self.n_v1, idx // self.n_v1
            y, x = pos // self.grid, pos % self.grid
            i, j = torch.triu_indices(len(idx), len(idx), 1)
            dy, dx = y[j] - y[i] + self.grid - 1, x[j] - x[i] + self.grid - 1
            out.append(((cell[i] * self.n_v1 + cell[j]) * self.span + dy) * self.span + dx)
        return out


class EpisodicMemory:
    """Hippocampus: for each action, a store of (key, return). Keys are random projections of the sparse code.
    Recall: exact match -> its value; otherwise the mean of the k most similar memories (pattern completion).
    Write: at the end of an episode every (state, action) stores its discounted return (keeping the best)."""

    def __init__(self, n_features, n_actions, capacity=20000, key_dim=32, k=7, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(n_features, key_dim, generator=g) / math.sqrt(key_dim)
        self.keys = torch.zeros(n_actions, capacity, key_dim)
        self.values = torch.zeros(n_actions, capacity)
        self.size = [0] * n_actions
        self.next = [0] * n_actions
        self.capacity, self.k, self.A = capacity, k, n_actions

    def key(self, codes):
        return torch.stack([self.proj[c].sum(0) for c in codes])                # [N, key_dim]

    def recall(self, keys):
        """Returns Q_ec [N, A] and a mask [N, A] of actions with at least k memories."""
        q = torch.zeros(keys.shape[0], self.A)
        known = torch.zeros(keys.shape[0], self.A, dtype=torch.bool)
        for a in range(self.A):
            n = self.size[a]
            if n < self.k:
                continue
            d2 = torch.cdist(keys, self.keys[a, :n])                            # [N, n]
            dist, idx = torch.topk(d2, self.k, largest=False)
            vals = self.values[a, :n][idx]
            exact = dist[:, 0] < 1e-4
            q[:, a] = torch.where(exact, vals[:, 0], vals.mean(1))
            known[:, a] = True
        return q, known

    def write(self, keys, actions, returns):
        for a in range(self.A):
            sel = actions == a
            if not sel.any():
                continue
            ks, rs = keys[sel], returns[sel]
            n = self.size[a]
            if n > 0:                                                            # exact match: keep the best return
                d, idx = torch.cdist(ks, self.keys[a, :n]).min(1)
                exact = d < 1e-4
                if exact.any():
                    j = idx[exact]
                    self.values[a, j] = torch.maximum(self.values[a, j], rs[exact])
                ks, rs = ks[~exact], rs[~exact]
            for key, ret in zip(ks, rs):                                         # new memories (oldest replaced)
                slot = self.next[a]
                self.keys[a, slot], self.values[a, slot] = key, ret
                self.next[a] = (slot + 1) % self.capacity
                self.size[a] = min(self.size[a] + 1, self.capacity)


class HippocampalPlanner:
    """
    World model + planning. The hippocampus keeps a map of experienced transitions (features, action, reward,
    next features, end of episode). Planning = replay through this map, prioritized by surprise |δ|
    (Mattar & Daw, 2018): each replayed transition trains the striatum toward r + γ max_a' Q(s', a'),
    so a newly discovered reward is propagated back along the paths that lead to it.
    """

    def __init__(self, capacity=50000, alpha=0.6, seed=0):
        self.capacity, self.alpha = capacity, alpha
        self.items = [None] * capacity
        self.priority = torch.zeros(capacity)
        self.size = self.next = 0
        self.gen = torch.Generator().manual_seed(seed + 7)

    def store(self, features, action, reward, next_features, terminal, surprise):
        self.items[self.next] = (features, action, reward, next_features, terminal)
        self.priority[self.next] = (abs(surprise) + 1e-3) ** self.alpha
        self.next = (self.next + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        if self.size == 0:
            return []
        idx = torch.multinomial(self.priority[:self.size], n, replacement=True, generator=self.gen)
        return [(int(i), self.items[int(i)]) for i in idx]


class VisualBrainAgent:
    """
    n_actions        discrete actions of the game
    lr, gamma, lam   striatal learning rate, discount, eligibility decay (tags kept for `trace_len` steps)
    temperature      softmax temperature of the action choice
    conjunctions     use dendritic conjunction features
    episodic         weight of the hippocampal values in the choice (0 = no hippocampus)
    """

    def __init__(self, n_actions, n_envs=8, lr=0.1, gamma=0.99, lam=0.8, trace_len=10, temperature=0.05,
                 conjunctions=True, episodic=0.0, ec_capacity=20000, planning=0, plan_lr=None,
                 critical_period=3000, seed=0):
        self.N, self.A = n_envs, n_actions
        self.lr, self.gamma, self.lam, self.trace_len, self.temperature = lr, gamma, lam, trace_len, temperature
        self.gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.vision = VisualCortex(n_envs, critical_period=critical_period, seed=seed)
        self.habituation = Habituation(self.vision.v1_size)
        self.conj = DendriticConjunctions(self.vision.n_v1) if conjunctions else None
        n_total = self.vision.n_features + (self.conj.size if self.conj else 0)
        self.w = torch.zeros(n_total, n_actions)                    # cortico-striatal synapses
        self.dopamine = Dopamine(n_envs, gamma)
        self.history = [[] for _ in range(n_envs)]                  # recent (features, action) = eligibility tags
        self.episodic = episodic
        if episodic > 0:
            self.memory = EpisodicMemory(self.vision.n_features, n_actions, ec_capacity, seed=seed)
            self.episode = [[] for _ in range(n_envs)]              # (key, action, raw reward) of the running episode
        self.prev_q = None
        self.planning = planning
        self.plan_lr = lr / 4 if plan_lr is None else plan_lr
        if planning > 0:
            self.planner = HippocampalPlanner(seed=seed)
            self.prev_features = None

    # -------------------------------------------------------------------------------- perception
    def perceive(self, frames, new_episode):
        codes, _ = self.vision(frames, new_episode)
        if self.conj is None:
            return codes, codes
        v1 = [c[c < self.vision.v1_size] for c in codes]
        moving = self.habituation(v1)
        pairs = self.conj(moving)
        offset = self.vision.n_features
        return [torch.cat([c, p + offset]) for c, p in zip(codes, pairs)], codes

    def q_values(self, features):
        return torch.stack([self.w[f].sum(0) for f in features])           # [N, A]

    def choose(self, q, ec_q=None, known=None):
        scores = q
        if ec_q is not None:                                                # arbitration striatum / hippocampus
            ec = ec_q / math.sqrt(self.dopamine.var + 1e-8)                 # same units as the striatum
            scores = torch.where(known, q + self.episodic * ec, q)
        probs = torch.softmax(scores / self.temperature, -1)
        return torch.multinomial(probs, 1, generator=self.gen)[:, 0]

    # -------------------------------------------------------------------------------- loop
    def reset(self, frames):
        new = torch.ones(self.N, dtype=torch.bool)
        return self._act(frames, new, None, None)

    def step(self, frames, reward, done):
        reward = torch.as_tensor(np.asarray(reward), dtype=torch.float32).reshape(self.N)
        done = torch.as_tensor(np.asarray(done), dtype=torch.bool).reshape(self.N)
        return self._act(frames, done, reward, self.dopamine(reward, done))

    def _act(self, frames, new, raw_reward, r):
        features, codes = self.perceive(frames, new)
        q = self.q_values(features)
        ec_q = known = keys = None
        if self.episodic > 0:
            keys = self.memory.key(codes)
            ec_q, known = self.memory.recall(keys)
        action = self.choose(q, ec_q, known)
        q_sa = q.gather(1, action[:, None])[:, 0]
        if r is not None:
            delta = r + self.gamma * q_sa * (~new) - self.prev_q          # dopamine: reward prediction error
            self._learn(delta)
            if self.planning > 0:
                for n in range(self.N):
                    self.planner.store(self.prev_features[n], int(self.prev_action[n]), float(r[n]),
                                       features[n], bool(new[n]), float(delta[n]))
                self._plan()
            if self.episodic > 0:
                self._remember(raw_reward, new)
        for n in range(self.N):                                            # tag the synapses of this choice
            if new[n]:
                self.history[n] = []
            self.history[n].insert(0, (features[n], int(action[n])))
            del self.history[n][self.trace_len:]
        if self.episodic > 0:
            for n in range(self.N):
                self.episode[n].append([keys[n], int(action[n]), 0.0])
        self.prev_q = self.q_values(features).gather(1, action[:, None])[:, 0]
        if self.planning > 0:
            self.prev_features, self.prev_action = features, action
        return action.numpy()

    def _plan(self):
        """Replay `planning` remembered transitions per environment step (mental simulation)."""
        batch = self.planner.sample(self.planning * self.N)
        rows, cols, vals = [], [], []
        for i, (f, a, r, f2, terminal) in batch:
            target = r + (0.0 if terminal else self.gamma * float(self.w[f2].sum(0).max()))
            delta = max(-1.0, min(1.0, target - float(self.w[f, a].sum())))   # replayed dopamine is bounded
            self.planner.priority[i] = (abs(delta) + 1e-3) ** self.planner.alpha
            rows.append(f); cols.append(torch.full_like(f, a)); vals.append(torch.full(f.shape, delta / len(f)))
        if rows:
            self.w.index_put_((torch.cat(rows), torch.cat(cols)), self.plan_lr * torch.cat(vals), accumulate=True)

    def _learn(self, delta):
        """Replacing eligibility tags: a synapse keeps only the tag of its most recent use, decayed by (γλ)^age.
        (Accumulating tags would move a value several times per step, because consecutive frames share most
        of their features; that is unstable.)"""
        rows, cols, vals = [], [], []
        decay = self.gamma * self.lam
        for n in range(self.N):
            if not self.history[n]:
                continue
            feats = torch.cat([f for f, _ in self.history[n]])
            ages = torch.cat([torch.full_like(f, age) for age, (f, _) in enumerate(self.history[n])])
            acts = torch.cat([torch.full_like(f, a) for f, a in self.history[n]])
            sizes = torch.cat([torch.full(f.shape, float(max(len(f), 1))) for f, _ in self.history[n]])
            order = torch.argsort(feats * self.trace_len + ages)             # per feature, newest use first
            feats, ages, acts, sizes = feats[order], ages[order], acts[order], sizes[order]
            first = torch.ones_like(feats, dtype=torch.bool)
            first[1:] = feats[1:] != feats[:-1]
            rows.append(feats[first]); cols.append(acts[first])
            vals.append(float(delta[n]) * decay ** ages[first].float() / sizes[first])
        if rows:
            self.w.index_put_((torch.cat(rows), torch.cat(cols)), self.lr * torch.cat(vals), accumulate=True)

    def _remember(self, raw_reward, new):
        for n in range(self.N):
            if self.episode[n]:
                self.episode[n][-1][2] = float(raw_reward[n])              # reward of the last action
            if new[n] and self.episode[n]:
                steps, ret = self.episode[n], 0.0
                returns = []
                for _, _, rw in reversed(steps):
                    ret = rw + self.gamma * ret
                    returns.append(ret)
                returns.reverse()
                self.memory.write(torch.stack([s[0] for s in steps]), torch.tensor([s[1] for s in steps]),
                                  torch.tensor(returns))
                self.episode[n] = []
