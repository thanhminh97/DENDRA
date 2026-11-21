"""
DENDRA-Brain: a reinforcement-learning agent built from brain mechanisms, learning online with local rules.

No backpropagation, no replay buffer, no PPO: every synapse changes at every time step using only signals
available where it is (its own activity, a decaying eligibility tag, and the dopamine broadcast).

    observation ─► sensory adaptation ─► temporal context ─► cortical columns ─► striatum ─► action
                   (gain control)        (time cells)        (sparse place code)  (action channels)
                                                                                     ▲
    reward ─────► dopamine (adaptive coding) ─► δ × eligibility tags ────────────────┘

  Sensory adaptation   z-scores every input channel with running statistics (gain control).
  Temporal context     leaky traces of the input at several time scales ("time cells"). Input − trace tells
                       how fast and which way the input changes, so hidden velocities become visible without
                       a trained recurrent network.
  Cortical columns     many small areas, each seeing a few random input channels (its receptive field).
                       Inside an area, units are place cells: a unit fires only when the input is inside its
                       field, and lateral inhibition lets at most k of them fire (sparse code). New units are
                       recruited for situations nothing responds to, and the least used units are recycled.
  Striatum             action channels: one group of channels per action dimension (somatotopy), one channel
                       per discrete action or per motor level. Channel values Q are read out from the
                       firing cortical units and learn with a dopamine-gated Sarsa(λ) rule (Morris et al.,
                       2006). Choice = softmax over channels, plus a bonus for actions rarely tried in the
                       current situation (novelty-driven exploration, Kakade & Dayan, 2002).
  Eligibility traces   a synapse that just took part in a choice stays tagged for a while (decay γλ);
                       dopamine arriving later turns the tag into a weight change (Yagishita et al., 2014).
                       This carries credit back over time without backpropagation through time.
  Dopamine             δ = r + γ Q(next) − Q(now), with rewards divided by the running spread of the
                       return (adaptive coding, Tobler et al., 2005) and clipped to a bounded range.

Usage (N environments in parallel; done[i] = True means obs[i] is the first observation of a new episode
and reward[i] was the last reward of the previous one):

    agent = BrainAgent(obs_dim=4, n_actions=2, n_envs=8)                        # discrete actions
    agent = BrainAgent(obs_dim=3, action_low=[-2], action_high=[2], n_envs=8)    # continuous actions
    action = agent.reset(obs)
    while True:
        obs, reward, done = env.step(action)
        action = agent.step(obs, reward, done)    # learn from (reward, done), then choose the next action
"""
import math

import numpy as np
import torch


class SensoryAdaptation:
    """Running mean and variance of every input channel -> z-scores clipped to [-clip, clip]."""

    def __init__(self, dim, clip=5.0):
        self.mean, self.var, self.count, self.clip = torch.zeros(dim), torch.ones(dim), 1e-4, clip

    def __call__(self, x):
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = batch_mean - self.mean, self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class TemporalContext:
    """Leaky traces of the input at several time scales. Output: [input, input - trace_1, input - trace_2, ...]."""

    def __init__(self, dim, n_envs, timescales=(2.0, 8.0)):
        self.timescales = tuple(timescales)
        self.traces = torch.zeros(len(self.timescales), n_envs, dim)
        self.out_dim = dim * (1 + len(self.timescales))

    def __call__(self, x, new_episode):
        self.traces = torch.where(new_episode[None, :, None], x[None], self.traces)   # no history at a new start
        features = [x] + [x - trace for trace in self.traces]
        for i, tau in enumerate(self.timescales):
            self.traces[i] += (x - self.traces[i]) / tau
        return torch.cat(features, -1)


class PlaceCells:
    """
    One cortical area. Unit i prefers input c_i and fires when the input u is within its receptive field
    (|u - c_i| <= field * radius). Lateral inhibition: at most the k closest units fire.
      recruitment  if no unit is within `radius`, a free unit is recruited with c = u
                   (when all are in use, the least used unit is recycled)
      usage        slow running average of how often each unit fires
    """

    def __init__(self, in_dim, n_units, k, radius, field):
        self.prototypes = torch.zeros(n_units, in_dim)
        self.n_units, self.k = n_units, k
        self.radius2 = radius ** 2 * in_dim
        self.field2 = self.radius2 * field ** 2
        self.recruited = 0
        self.usage = torch.zeros(n_units)
        self.recycled = []

    def _distances(self, u):
        c = self.prototypes
        d2 = (u * u).sum(-1, keepdim=True) - 2 * u @ c.T + (c * c).sum(-1)[None]
        d2[:, self.recruited:] = float("inf")                   # units not yet recruited never fire
        return d2

    def __call__(self, u):
        self.recycled = []
        d2 = self._distances(u)
        for n in torch.nonzero(d2.min(-1).values > self.radius2).flatten().tolist():
            if self.recycled and (self._distances(u[n:n + 1]).min() <= self.radius2).item():
                continue                                        # already covered by a unit recruited just now
            unit = self.recruited if self.recruited < self.n_units else int(torch.argmin(self.usage))
            self.recruited = min(self.recruited + 1, self.n_units)
            self.prototypes[unit], self.usage[unit] = u[n], 1.0 / self.n_units
            self.recycled.append(unit)
        if self.recycled:
            d2 = self._distances(u)
        neg_d2, active = torch.topk(-d2, min(self.k, self.recruited), dim=-1)
        firing = (-neg_d2 <= self.field2).float()
        firing[:, 0] = 1.0                                      # the closest unit always fires
        fired = torch.zeros(self.n_units).index_add_(0, active.flatten(), firing.flatten())
        self.usage = 0.999 * self.usage + 0.001 * fired / u.shape[0]
        return active, firing


class Cortex:
    """Many small areas (columns), each with its own receptive field: a few random input channels.
    Returns the indices of the candidate units [N, areas*k] and whether each one fires (0/1)."""

    def __init__(self, in_dim, areas, subspace, n_units, k, radius, field, seed, primary=None):
        g = np.random.default_rng(seed)
        subspace = min(subspace, in_dim)
        uncovered, self.fields = list(g.permutation(in_dim)), []
        if primary:                                             # a primary area that sees the whole current input
            self.fields.append(list(primary))
            uncovered = [c for c in uncovered if c not in primary]
        while len(self.fields) < areas:
            chosen = [uncovered.pop()] if uncovered else []     # every channel is seen by at least one area
            others = [c for c in g.permutation(in_dim) if c not in chosen]
            self.fields.append(sorted(chosen + others[:subspace - len(chosen)]))
        self.areas = [PlaceCells(len(f), n_units, k, radius, field) for f in self.fields]
        self.n_units, self.units_per_area = n_units * areas, n_units
        self.recycled = []

    @property
    def recruited(self):
        return sum(area.recruited for area in self.areas)

    def __call__(self, u):
        active, firing, self.recycled = [], [], []
        for i, (field, area) in enumerate(zip(self.fields, self.areas)):
            a, f = area(u[:, field])
            active.append(a + i * self.units_per_area); firing.append(f)
            self.recycled += [unit + i * self.units_per_area for unit in area.recycled]
        return torch.cat(active, -1), torch.cat(firing, -1)


class Dopamine:
    """Reward -> teaching signal. Adaptive coding: rewards are divided by the running spread of the discounted
    return, so the same learning rates work for rewards of +1 (CartPole) or -16 (Pendulum) per step."""

    def __init__(self, n_envs, gamma):
        self.gamma = gamma
        self.returns = torch.zeros(n_envs)
        self.mean, self.var, self.count = 0.0, 1.0, 1.0         # prior: one step with spread 1

    def __call__(self, reward, done):
        self.returns = self.returns * self.gamma + reward
        m, v, n = float(self.returns.mean()), float(self.returns.var(unbiased=False)), reward.shape[0]
        delta, total = m - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / total) / total
        self.count = total
        self.returns[done] = 0
        return (reward / math.sqrt(self.var + 1e-8)).clamp(-10, 10)


class BrainAgent:
    """
    Discrete actions: give n_actions. Continuous actions: give action_low / action_high (each dimension gets
    `levels` motor channels spread evenly over its range).

    areas, subspace   number of cortical columns and input channels per column
    primary_max       if the observation has at most this many channels, one column sees all of it
    n_units, k        place cells per column and the most that may fire at once
    radius, field     recruitment radius and receptive-field size (input in z-score units, per channel)
    timescales        time constants of the temporal context (empty = no temporal context)
    gamma, lam        discount (how far ahead) and eligibility-tag decay
    lr                learning rate of the cortico-striatal synapses
    temperature       softmax temperature of the channel choice
    explore           bonus for actions rarely tried in the current situation (0 = off)
    adaptive          divide rewards by the running spread of the return (adaptive dopamine coding)
    novelty           novelty dopamine: intrinsic reward novelty / sqrt(1 + visits) of the least visited firing
                      cells, added to the reward so that it propagates to the states that lead there (0 = off)
    consolidation     metaplasticity: a synapse that has been used n times learns at rate
                      lr / sqrt(1 + consolidation * n), but never below lr * min_plasticity (0 = off)
    """

    def __init__(self, obs_dim, n_actions=None, action_low=None, action_high=None, n_envs=1,
                 areas=16, subspace=4, n_units=1024, k=4, radius=0.3, field=2.0, timescales=(2.0, 8.0),
                 gamma=0.99, lam=0.9, lr=0.2, temperature=0.01, explore=0.3, levels=5, primary_max=8,
                 adaptive=True, consolidation=0.0, min_plasticity=0.1, novelty=0.0, seed=0):
        self.N, self.gamma, self.lam, self.lr = n_envs, gamma, lam, lr
        self.temperature, self.explore = temperature, explore
        self.consolidation, self.min_plasticity = consolidation, min_plasticity
        self.novelty = novelty
        self.gen = torch.Generator().manual_seed(seed)
        self.context = TemporalContext(obs_dim, n_envs, timescales)
        self.senses = SensoryAdaptation(self.context.out_dim)
        primary = list(range(obs_dim)) if obs_dim <= primary_max else None
        self.cortex = Cortex(self.context.out_dim, areas, subspace, n_units, k, radius, field, seed, primary)
        self.dopamine = Dopamine(n_envs, gamma) if adaptive else (lambda reward, done: reward)
        self.continuous = n_actions is None
        if self.continuous:
            low = torch.as_tensor(np.asarray(action_low), dtype=torch.float32)
            high = torch.as_tensor(np.asarray(action_high), dtype=torch.float32)
            self.motor_levels = low[:, None] + (high - low)[:, None] * torch.linspace(0, 1, levels)[None]   # [D, L]
            self.D, self.L = low.shape[0], levels
        else:
            self.D, self.L = 1, n_actions
        U = self.cortex.n_units
        self.visits = torch.zeros(U)                             # how often each place cell has fired
        self.w = torch.zeros(U, self.D, self.L)                  # cortico-striatal synapses
        self.tags = torch.zeros(n_envs, U, self.D, self.L)       # eligibility tags
        self.tried = torch.zeros(U, self.D, self.L)              # how often each channel was chosen from each unit
        self.rows = torch.arange(n_envs)[:, None]
        self.prev = None

    # -------------------------------------------------------------------------------- perception
    def perceive(self, obs, new_episode):
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32).reshape(self.N, -1)
        active, firing = self.cortex(self.senses(self.context(x, new_episode)))
        for unit in self.cortex.recycled:                        # a recycled unit starts with blank synapses
            self.w[unit] = 0; self.tags[:, unit] = 0; self.tried[unit] = 0; self.visits[unit] = 0
        return active, firing

    # -------------------------------------------------------------------------------- striatum
    def channel_values(self, active, firing):
        """Q of every channel: sum over the firing units. [N, D, L]"""
        return (self.w[active] * firing[..., None, None]).sum(1)

    def choose(self, q, active, firing):
        scores = q
        if self.explore > 0:                                     # novelty of each action in this situation
            tried = (self.tried[active] * firing[..., None, None]).sum(1) / firing.sum(1)[:, None, None]
            scores = q + self.explore / torch.sqrt(1.0 + tried)
        probs = torch.softmax(scores / self.temperature, -1).reshape(-1, self.L)
        choice = torch.multinomial(probs, 1, generator=self.gen).reshape(self.N, self.D)   # [N, D]
        for d in range(self.D):
            self.tried[:, d].index_put_((active.flatten(), choice[:, d:d + 1].expand_as(active).flatten()),
                                        firing.flatten(), accumulate=True)
        return choice

    def value_of(self, q, choice):
        """Value of the chosen action: mean over action dimensions of the chosen channels' Q. [N]"""
        return q.gather(-1, choice[..., None])[..., 0].mean(-1)

    def tag(self, active, firing, choice):
        """Replacing eligibility tags: each firing unit tags the chosen channel of every action dimension."""
        rows = self.rows.expand_as(active)
        old = self.tags[rows, active]                            # [N, k', D, L]
        kept = old * (1 - firing[..., None, None])               # firing units drop the tags of other channels
        index = choice[:, None, :, None].expand(-1, active.shape[1], -1, 1)
        chosen = torch.zeros_like(kept).scatter_(-1, index, firing[..., None, None].expand(-1, -1, self.D, 1))
        self.tags[rows, active] = torch.maximum(kept, chosen)

    def to_env_action(self, choice):
        if self.continuous:
            return self.motor_levels.gather(1, choice.T).T.numpy()       # [N, D] motor command
        return choice[:, 0].numpy()

    # -------------------------------------------------------------------------------- the loop
    def reset(self, obs):
        new = torch.ones(self.N, dtype=torch.bool)
        active, firing = self.perceive(obs, new)
        q = self.channel_values(active, firing)
        choice = self.choose(q, active, firing)
        self.tags.zero_()
        self.tag(active, firing, choice)
        self.prev = self.value_of(q, choice)
        return self.to_env_action(choice)

    def step(self, obs, reward, done):
        reward = torch.as_tensor(np.asarray(reward), dtype=torch.float32).reshape(self.N)
        done = torch.as_tensor(np.asarray(done), dtype=torch.bool).reshape(self.N)
        r = self.dopamine(reward, done)
        active, firing = self.perceive(obs, done)
        if self.novelty > 0:                                     # novelty dopamine (least visited firing cells)
            novel = torch.where(firing > 0, 1.0 / torch.sqrt(1.0 + self.visits[active]), torch.zeros_like(firing))
            r = r + self.novelty * torch.topk(novel, 4, dim=-1).values.mean(-1)
            self.visits.index_add_(0, active.flatten(), firing.flatten())
        q = self.channel_values(active, firing)
        choice = self.choose(q, active, firing)
        delta = r + self.gamma * self.value_of(q, choice) * (~done) - self.prev       # dopamine: prediction error
        k = active.shape[1]
        change = torch.einsum("n,nudl->udl", delta, self.tags)
        if self.consolidation > 0:                               # well-used synapses are consolidated: less plastic
            change *= (1.0 / torch.sqrt(1.0 + self.consolidation * self.tried)).clamp(min=self.min_plasticity)
        self.w += (self.lr / k) * change
        self.tags *= self.gamma * self.lam
        self.tags[done] = 0
        self.tag(active, firing, choice)
        self.prev = self.value_of(self.channel_values(active, firing), choice)       # after this step's learning
        return self.to_env_action(choice)
