"""
DENDRA for reinforcement learning: a recurrent memory core and a ready-made actor-critic agent,
for discrete actions (Atari, MinAtar, ...) and continuous actions (MuJoCo, Pendulum, ...).

    from dendra.rl import DendraAgent
    agent = DendraAgent(obs_dim=8, n_actions=4)                          # discrete
    agent = DendraAgent(obs_dim=17, n_actions=6, continuous=True)        # continuous
    state = agent.init_state(n_envs)
    dist, value, state = agent.step(obs, state, done)                    # one step for n_envs environments
    action = dist.sample()

The memory of DENDRA is the network state (Hebbian synapses + neuromodulator baseline), not a list of
past observations, so one step costs the same at any episode length. A whole rollout segment can also be
re-run in parallel over time with seq(), which gives the same result as calling step() T times.
"""
import math

import torch
import torch.nn as nn

from .model import CorticalRegion, Neuromodulators


def init_layer(layer, std=math.sqrt(2)):
    """Orthogonal weights and zero bias: the standard initialization for PPO networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class DendraCore(nn.Module):
    """
    DENDRA as an RL memory: a stack of cortical regions (spiking dendritic E/I neurons + Hebbian synapses)
    modulated by neuromodulators. Input: a feature vector per environment; output: a vector of size d.

    Its memory (a dict returned by init_state) has three parts:
      S  [regions, N, heads, dh, dh]  Hebbian synaptic weights of each region (the fast-weight memory)
      m  [N, d]                       running average of the input, read by the neuromodulators
      t  [N]                          steps since the episode began (to normalize that average)
    `done[i] = 1` means environment i starts a new episode at this step: its memory is cleared first.
    """

    def __init__(self, in_dim, d=128, regions=2, heads=4):
        super().__init__()
        self.inp = init_layer(nn.Linear(in_dim, d), 1.0)
        self.regions = nn.ModuleList([CorticalRegion(d, heads, int(1.5 * d), 4, 4) for _ in range(regions)])
        self.nm = Neuromodulators(d, regions, heads)
        self.n_regions, self.n_heads, self.d = regions, heads, d
        self.rates = []                                    # firing rates, collected for the homeostasis loss

    def init_state(self, n, device=None):
        """Empty memory for n environments."""
        device = device or self.inp.weight.device
        dh = self.d // self.n_heads
        return {"S": torch.zeros(self.n_regions, n, self.n_heads, dh, dh, device=device),
                "m": torch.zeros(n, self.d, device=device),
                "t": torch.zeros(n, device=device)}

    def neuromodulator_signals(self, m, t):
        """From the running input average m (after t+1 steps), compute for each region:
        gain (noradrenaline: how excitable the neurons are) and ach (acetylcholine: how strongly to write memory)."""
        decay = self.nm.decay
        normalizer = (1 - decay ** (t + 1)) / (1 - decay)  # sum of the averaging weights so far
        z = 2 * torch.sigmoid(self.nm.proj(self.nm.norm(m / normalizer[..., None])))
        z = z.view(*m.shape[:-1], self.n_regions, 1 + self.n_heads)
        return z[..., :1], z[..., 1:]                      # gain [..., L, 1], ach [..., L, heads]

    def step(self, x, state, done=None):
        """One time step. x [N, in_dim]; done [N] or None. Returns (output [N, d], new state)."""
        if done is None:
            done = torch.zeros(x.shape[0], device=x.device)
        keep = 1 - done
        S = state["S"] * keep[None, :, None, None, None]
        m, t = state["m"] * keep[:, None], state["t"] * keep
        x = self.inp(x)
        m = self.nm.decay * m + x
        gain, ach = self.neuromodulator_signals(m, t)
        h, new_S = x[:, None], []                          # [N, 1, d]: a sequence of length 1
        for l, region in enumerate(self.regions):
            h, S_l = region.step(h, gain[:, None, l], ach[:, None, l], None, S[l])
            new_S.append(S_l)
        self.rates.append(torch.stack([region.l5.last_rate for region in self.regions]))
        return h[:, 0], {"S": torch.stack(new_S), "m": m, "t": t + 1}

    def seq(self, x, state, done=None):
        """
        A whole segment of T steps at once (parallel over time). Same result as calling step() T times,
        but much faster for training. x [T, N, in_dim]; done [T, N] or None. Returns (output [T, N, d], new state).

        The Hebbian memory has a closed form. Inside one episode, the output at step i is
            o_i = sum_{j<=i} lam^(i-j) * w_j * (q_i . k_j) * v_j   +   lam^(i+1) * S0 q_i
        where the second term (the memory carried in from before the segment) only counts
        while no episode boundary has happened yet.
        """
        if done is None:
            done = torch.zeros(x.shape[:2], device=x.device)
        T, N = done.shape
        x = self.inp(x)                                                       # [T, N, d]

        # Neuromodulators: a cheap running average, computed step by step
        m, t, all_m, all_t = state["m"], state["t"], [], []
        for i in range(T):
            keep = 1 - done[i]
            m = self.nm.decay * m * keep[:, None] + x[i]
            t = t * keep
            all_m.append(m); all_t.append(t)
            t = t + 1
        all_m, all_t = torch.stack(all_m), torch.stack(all_t)                 # [T, N, d], [T, N]
        gain, ach = self.neuromodulator_signals(all_m, all_t)                # [T, N, L, .]
        gain, ach = gain.transpose(0, 1), ach.transpose(0, 1)                # [N, T, L, .]

        # Episode bookkeeping
        episode_id = done.transpose(0, 1).cumsum(1)                          # [N, T] resets seen up to step t
        same_episode = episode_id[:, :, None] == episode_id[:, None, :]      # [N, T, T] steps i and j in one episode
        before_first_reset = episode_id == 0                                 # [N, T] still sees the carried-in memory
        in_last_episode = episode_id == episode_id[:, -1:]                   # [N, T] writes into the final memory
        steps = torch.arange(T, device=x.device)
        distance = (steps[:, None] - steps[None, :]).float()                 # [T, T] i - j

        h, new_S, rates = x.transpose(0, 1), [], []                          # h [N, T, d]
        for l, region in enumerate(self.regions):
            memory = region.l23                                               # HebbianSTDP of this region
            x_in = region.l4a(h)
            q, k, v, w = memory._proj(x_in, ach[:, :, l])                    # q,k,v [N,T,heads,dh]; w [N,T,heads]
            log_lam = torch.log(memory.lam())                                 # [heads] per-head decay

            # Within the segment: causal STDP window lam^(i-j), limited to the same episode
            window = torch.where(distance >= 0, torch.exp(log_lam[:, None, None] * distance.clamp(min=0)),
                                 torch.zeros((), device=x.device))            # [heads, T, T]
            window = window[None] * same_episode[:, None]                     # [N, heads, T, T]
            scores = torch.einsum("bihd,bjhd->bhij", q, k) * window * w.permute(0, 2, 1)[:, :, None, :]
            out = torch.einsum("bhij,bjhd->bihd", scores, v)

            # Memory carried in from before the segment, read with decay lam^(i+1)
            S0 = state["S"][l]                                                # [N, heads, dh, dh]
            decay_in = torch.exp(log_lam[None, :] * (steps[:, None] + 1).float())                 # [T, heads]
            out = out + torch.einsum("bhde,bihe->bihd", S0, q) * (decay_in[None, :, :, None] * before_first_reset[:, :, None, None])

            # Memory left at the end of the segment (only steps of the last episode write into it)
            decay_out = torch.exp(log_lam[None, :] * (T - 1 - steps)[:, None].float())            # [T, heads]
            weighted_v = v * (w * decay_out[None] * in_last_episode[:, :, None])[..., None]
            S_end = (torch.exp(log_lam * T)[None, :, None, None] * S0 * (episode_id[:, -1] == 0)[:, None, None, None]
                     + torch.einsum("bjhd,bjhe->bhde", weighted_v, k))

            # The rest of the region: memory output, then the spiking dendritic layer (L5)
            h = h + memory._finish(x_in, out)
            h = h + region.l5(region.l4b(h), gain[:, :, l])
            new_S.append(S_end)
            rates.append(region.l5.last_rate)
        self.rates.append(torch.stack(rates))
        return h.transpose(0, 1), {"S": torch.stack(new_S), "m": all_m[-1], "t": all_t[-1] + 1}

    def clear_rates(self):
        """Forget the firing rates collected so far (call before the forward pass you train on)."""
        self.rates.clear()

    def homeostasis_loss(self, target=0.15):
        """Keeps the firing rate of every region near `target`, like homeostatic plasticity in the brain.
        Uses the rates collected by step()/seq() since the last clear_rates(); add it to the RL loss."""
        rates = torch.stack(self.rates).mean(0)
        return ((rates - target) ** 2).sum()


class DendraAgent(nn.Module):
    """
    Ready-made actor-critic with a DENDRA memory, for discrete or continuous actions:
        observation -> encoder -> DendraCore -> policy distribution + value

    obs_dim     size of a (flat) observation; ignored when you pass your own `encoder`
    n_actions   number of actions (discrete) or action dimensions (continuous)
    continuous  False: Categorical policy; True: Gaussian policy with a learned std
    encoder     optional module for images etc.; it must output vectors of size `encoder_dim`
    """

    def __init__(self, obs_dim, n_actions, continuous=False, d=128, regions=2, heads=4,
                 encoder=None, encoder_dim=None):
        super().__init__()
        self.continuous = continuous
        if encoder is None:                                # small MLP for vector observations
            encoder = nn.Sequential(init_layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                    init_layer(nn.Linear(64, 64)), nn.Tanh())
            encoder_dim = 64
        self.encoder = encoder
        self.core = DendraCore(encoder_dim, d, regions, heads)
        self.policy_head = init_layer(nn.Linear(d, n_actions), 0.01)
        self.value_head = init_layer(nn.Linear(d, 1), 1.0)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(n_actions))

    def init_state(self, n, device=None):
        return self.core.init_state(n, device)

    def distribution(self, h):
        if self.continuous:
            return torch.distributions.Normal(self.policy_head(h), self.log_std.exp())
        return torch.distributions.Categorical(logits=self.policy_head(h))

    def step(self, obs, state, done=None):
        """One step for N environments. obs [N, ...] -> (distribution, value [N], new state)."""
        h, state = self.core.step(self.encoder(obs), state, done)
        return self.distribution(h), self.value_head(h)[:, 0], state

    def seq(self, obs, state, done=None):
        """A whole segment, parallel over time. obs [T, N, ...] -> (distribution, value [T, N], new state)."""
        T, N = obs.shape[:2]
        features = self.encoder(obs.flatten(0, 1)).view(T, N, -1)
        h, state = self.core.seq(features, state, done)
        return self.distribution(h), self.value_head(h)[..., 0], state

    def log_prob(self, dist, action):
        """Log-probability of an action (summed over dimensions for continuous actions)."""
        logp = dist.log_prob(action)
        return logp.sum(-1) if self.continuous else logp
