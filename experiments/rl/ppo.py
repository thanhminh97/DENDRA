"""
Recurrent PPO with a swappable memory "core", for DISCRETE actions (Atari, MinAtar)
and CONTINUOUS actions (Pendulum, MuJoCo).

The agent is   observation -> Encoder -> Core (memory) -> policy head + value head.
Only the core changes between runs:
  dendra   DENDRA: Hebbian synapses + spiking dendritic E/I neurons + neuromodulators
  lstm     an LSTM cell (the usual choice for recurrent RL)
  mlp      no memory at all (it sees the last 4 frames instead, the usual choice for Atari)

Environments (--env):
  atari:Breakout          real Atari 2600 (ALE), 84x84 grayscale, frame skip 4
  minatar:breakout        miniature 10x10 Atari (MinAtar), fast on CPU
  pendulum-pomdp          Pendulum with the angular velocity HIDDEN, so it must be inferred from memory
  mujoco:HalfCheetah-v5   full MuJoCo; add the suffix "-pomdp" to keep positions only (velocities hidden)

Example:
  python experiments/rl/ppo.py --env minatar:breakout --core dendra --total 300000 --out runs/ppo.json

File layout
  1. Settings (command line)
  2. Environments (vectorized, reset automatically at the end of an episode)
  3. Networks: encoder, the three cores, the actor-critic agent
  4. PPO: reward normalization, rollout collection, advantages (GAE), policy update
  5. Main training loop and saving of results
"""
import argparse
import json
import math
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dendra.model import CorticalRegion, Neuromodulators


# ============================================================================ 1. Settings
parser = argparse.ArgumentParser(description="Recurrent PPO with a DENDRA, LSTM or memoryless core")
parser.add_argument("--env", default="minatar:breakout", help="environment, see the list at the top of this file")
parser.add_argument("--core", default="dendra", choices=["dendra", "lstm", "mlp"], help="memory core of the agent")
parser.add_argument("--total", type=int, default=300_000, help="total environment steps")
parser.add_argument("--num_envs", type=int, default=8, help="environments run in parallel")
parser.add_argument("--num_steps", type=int, default=128, help="steps per environment between two updates")
parser.add_argument("--minibatches", type=int, default=4, help="minibatches per epoch (split by environment)")
parser.add_argument("--epochs", type=int, default=None, help="PPO epochs per update; default 4 (discrete), 10 (continuous)")
parser.add_argument("--lr", type=float, default=None, help="learning rate; default 2.5e-4 (discrete), 3e-4 (continuous)")
parser.add_argument("--d", type=int, default=128, help="width of the core")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_minutes", type=float, default=None, help="stop early after this many minutes (results are still saved)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--out", default="runs/ppo.json", help="where to write the JSON results")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(int(os.environ.get("NT", "1")))   # several runs share the CPU, so one thread each by default

DEVICE = torch.device(args.device)
ENV_FAMILY = args.env.split(":")[0]                      # "atari", "minatar", "pendulum-pomdp" or "mujoco"
CONTINUOUS = ENV_FAMILY in ("pendulum-pomdp", "mujoco")  # continuous actions -> Gaussian policy


# ============================================================================ 2. Environments
class MinAtarVec:
    """Several MinAtar games side by side. Resets a game as soon as it ends. Observations: [N, C, 10, 10]."""

    def __init__(self, game, n, seed):
        from minatar import Environment
        self.envs = [Environment(game) for _ in range(n)]
        for i, env in enumerate(self.envs):
            env.seed(seed * 1000 + i)   # seeds both the sticky-action and the game's random stream (reproducible runs)
            env.reset()
        self.n_actions = self.envs[0].num_actions()
        self.obs_shape = (self.envs[0].state_shape()[2], 10, 10)

    def observe(self):
        return np.stack([env.state().transpose(2, 0, 1) for env in self.envs]).astype(np.float32)

    def reset(self):
        return self.observe()

    def step(self, actions):
        rewards = np.zeros(len(self.envs), np.float32)
        dones = np.zeros(len(self.envs), np.float32)
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            reward, done = env.act(int(action))
            rewards[i], dones[i] = reward, float(done)
            if done:
                env.reset()
        return self.observe(), rewards, dones


class GymVec:
    """Gymnasium environments side by side. A finished episode is reset within the same step,
    so the returned observation is already the first one of the next episode."""

    def __init__(self, make_env_fns, seed):
        self.vec = gym.vector.SyncVectorEnv(make_env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.seed = seed
        space = self.vec.single_action_space
        self.n_actions = space.n if hasattr(space, "n") else space.shape[0]
        self.obs_shape = self.vec.single_observation_space.shape
        if CONTINUOUS:
            self.low, self.high = space.low, space.high

    def reset(self):
        return self.vec.reset(seed=self.seed)[0].astype(np.float32)

    def step(self, actions):
        if CONTINUOUS:
            actions = np.clip(actions, self.low, self.high)
        obs, rewards, terminated, truncated, _ = self.vec.step(actions)
        dones = terminated | truncated
        return obs.astype(np.float32), rewards.astype(np.float32), dones.astype(np.float32)


def keep_first_values(env, k, low, high):
    """Observation wrapper that keeps only the first k numbers (used to hide velocities)."""
    return gym.wrappers.TransformObservation(env, lambda obs: obs[:k].astype(np.float32),
                                             gym.spaces.Box(low, high, (k,), np.float32))


def stack_vector_frames(env, frame_stack):
    """Give a vector observation the last `frame_stack` steps, flattened into one vector."""
    if frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        env = gym.wrappers.FlattenObservation(env)
    return env


def make_atari(game, frame_stack):
    env = gym.make(f"ALE/{game}-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
    return gym.wrappers.FrameStackObservation(env, frame_stack)


def make_pendulum_pomdp(frame_stack):
    env = gym.make("Pendulum-v1")
    env = keep_first_values(env, 2, -1.0, 1.0)             # keep cos(angle), sin(angle); drop angular velocity
    return stack_vector_frames(env, frame_stack)


def make_mujoco(name, frame_stack):
    hide_velocities = name.endswith("-pomdp")
    env = gym.make(name.replace("-pomdp", ""))
    if hide_velocities:
        n_positions = env.unwrapped.model.nq - 1           # joint positions in the observation; velocities follow them
        env = keep_first_values(env, n_positions, -np.inf, np.inf)
    return stack_vector_frames(env, frame_stack)


def make_vec_env(frame_stack):
    n, seed = args.num_envs, args.seed
    if ENV_FAMILY == "minatar":
        return MinAtarVec(args.env.split(":")[1], n, seed)
    if ENV_FAMILY == "atari":
        import ale_py
        gym.register_envs(ale_py)
        game = args.env.split(":")[1]
        return GymVec([lambda: make_atari(game, frame_stack) for _ in range(n)], seed)
    if ENV_FAMILY == "pendulum-pomdp":
        return GymVec([lambda: make_pendulum_pomdp(frame_stack) for _ in range(n)], seed)
    if ENV_FAMILY == "mujoco":
        name = args.env.split(":")[1]
        return GymVec([lambda: make_mujoco(name, frame_stack) for _ in range(n)], seed)
    raise ValueError(f"unknown environment: {args.env}")


# ============================================================================ 3. Networks
def init_layer(layer, std=math.sqrt(2)):
    """Orthogonal weights and zero bias: the standard PPO initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(nn.Module):
    """Turns one observation into a feature vector: a CNN for images, a small MLP for vectors."""

    def __init__(self, obs_shape):
        super().__init__()
        is_image = len(obs_shape) == 3
        if is_image and obs_shape[-1] == 84:              # Atari, 84x84 (the "Nature DQN" CNN)
            self.net = nn.Sequential(
                init_layer(nn.Conv2d(obs_shape[0], 32, 8, 4)), nn.ReLU(),
                init_layer(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(),
                nn.Flatten(), init_layer(nn.Linear(3136, 512)), nn.ReLU())
            self.dim, self.scale = 512, 1 / 255.0
        elif is_image:                                     # MinAtar, 10x10
            self.net = nn.Sequential(
                init_layer(nn.Conv2d(obs_shape[0], 16, 3, 1)), nn.ReLU(),
                nn.Flatten(), init_layer(nn.Linear(16 * 8 * 8, 128)), nn.ReLU())
            self.dim, self.scale = 128, 1.0
        else:                                              # vector observation
            self.net = nn.Sequential(
                init_layer(nn.Linear(int(np.prod(obs_shape)), 64)), nn.Tanh(),
                init_layer(nn.Linear(64, 64)), nn.Tanh())
            self.dim, self.scale = 64, 1.0

    def forward(self, obs):
        return self.net(obs * self.scale)


# Every core has the same interface:
#   init_state(n)            -> memory for n environments (a dict of tensors)
#   step(x, state, done)     -> (output, new state) for one time step; done[i] = 1 clears environment i's memory first
#   seq(x, state, done)      -> optional: a whole segment of T steps at once, same result as T calls to step()

class MLPCore(nn.Module):
    """No memory: one dense layer."""

    def __init__(self, in_dim, d):
        super().__init__()
        self.fc = init_layer(nn.Linear(in_dim, d))

    def init_state(self, n):
        return {}

    def step(self, x, state, done):
        out = torch.tanh(self.fc(x)) if CONTINUOUS else F.relu(self.fc(x))
        return out, state


class LSTMCore(nn.Module):
    """Standard LSTM cell."""

    def __init__(self, in_dim, d):
        super().__init__()
        self.cell = nn.LSTMCell(in_dim, d)

    def init_state(self, n):
        zeros = torch.zeros(n, self.cell.hidden_size, device=DEVICE)
        return {"h": zeros, "c": zeros.clone()}

    def step(self, x, state, done):
        keep = (1 - done)[:, None]                         # 0 for environments that start a new episode
        h, c = self.cell(x, (state["h"] * keep, state["c"] * keep))
        return h, {"h": h, "c": c}


class DendraCore(nn.Module):
    """
    DENDRA as an RL memory: a stack of cortical regions (spiking dendritic E/I neurons + Hebbian synapses)
    modulated by neuromodulators. Its memory has three parts:
      S  [regions, N, heads, dh, dh]  Hebbian synaptic weights of each region (the fast-weight memory)
      m  [N, d]                       running average of the input, read by the neuromodulators
      t  [N]                          steps since the episode began (to normalize that average)
    All three are cleared when a new episode starts.
    """

    def __init__(self, in_dim, d, regions=2, heads=4):
        super().__init__()
        self.inp = init_layer(nn.Linear(in_dim, d), 1.0)
        self.regions = nn.ModuleList([CorticalRegion(d, heads, int(1.5 * d), 4, 4) for _ in range(regions)])
        self.nm = Neuromodulators(d, regions, heads)
        self.n_regions, self.n_heads, self.d = regions, heads, d
        self.rates = []                                    # firing rates, collected for the homeostasis loss

    def init_state(self, n):
        dh = self.d // self.n_heads
        return {"S": torch.zeros(self.n_regions, n, self.n_heads, dh, dh, device=DEVICE),
                "m": torch.zeros(n, self.d, device=DEVICE),
                "t": torch.zeros(n, device=DEVICE)}

    def neuromodulator_signals(self, m, t):
        """From the running input average m (after t+1 steps), compute for each region:
        gain (noradrenaline: how excitable the neurons are) and ach (acetylcholine: how strongly to write memory)."""
        decay = self.nm.decay
        normalizer = (1 - decay ** (t + 1)) / (1 - decay)  # sum of the averaging weights so far
        z = 2 * torch.sigmoid(self.nm.proj(self.nm.norm(m / normalizer[..., None])))
        z = z.view(*m.shape[:-1], self.n_regions, 1 + self.n_heads)
        return z[..., :1], z[..., 1:]                      # gain [..., L, 1], ach [..., L, heads]

    def step(self, x, state, done):
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

    def seq(self, x, state, done):
        """
        A whole segment of T steps at once (parallel over time). Same result as calling step() T times,
        but much faster for training. x [T, N, F]; done [T, N] (1 = this step starts a new episode).

        The Hebbian memory has a closed form. Inside one episode, the output at step i is
            o_i = sum_{j<=i} lam^(i-j) * w_j * (q_i . k_j) * v_j   +   lam^(i+1) * S0 q_i
        where the second term (the memory carried in from before the segment) only counts
        while no episode boundary has happened yet.
        """
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


CORES = {"dendra": DendraCore, "lstm": LSTMCore, "mlp": MLPCore}


class Agent(nn.Module):
    """Actor-critic: encoder -> core -> (policy distribution, value)."""

    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.encoder = Encoder(obs_shape)
        self.core = CORES[args.core](self.encoder.dim, args.d)
        self.policy_head = init_layer(nn.Linear(args.d, n_actions), 0.01)
        self.value_head = init_layer(nn.Linear(args.d, 1), 1.0)
        if CONTINUOUS:
            self.log_std = nn.Parameter(torch.zeros(n_actions))

    def distribution(self, h):
        if CONTINUOUS:
            return torch.distributions.Normal(self.policy_head(h), self.log_std.exp())
        return torch.distributions.Categorical(logits=self.policy_head(h))

    def step(self, obs, state, done):
        """One step for N environments: obs [N, ...]."""
        h, state = self.core.step(self.encoder(obs), state, done)
        return self.distribution(h), self.value_head(h)[:, 0], state

    def seq(self, obs, state, done):
        """A whole segment: obs [T, N, ...]. DENDRA runs in parallel over time; LSTM/MLP loop step by step."""
        T, N = done.shape
        features = self.encoder(obs.flatten(0, 1)).view(T, N, -1)
        if hasattr(self.core, "seq"):
            h, state = self.core.seq(features, state, done)
        else:
            outputs = []
            for t in range(T):
                h_t, state = self.core.step(features[t], state, done[t])
                outputs.append(h_t)
            h = torch.stack(outputs)
        return self.distribution(h), self.value_head(h)[..., 0], state


# ============================================================================ 4. PPO
class RewardNormalizer:
    """Divides rewards by the running std of the discounted return (standard for continuous control)."""

    def __init__(self, n_envs, gamma):
        self.returns = np.zeros(n_envs)
        self.gamma = gamma
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def __call__(self, rewards, dones):
        self.returns = self.returns * self.gamma + rewards
        batch_mean, batch_var, batch_count = self.returns.mean(), self.returns.var(), len(rewards)
        delta, total = batch_mean - self.mean, self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total) / total
        self.count = total
        self.returns[dones > 0] = 0
        return np.clip(rewards / math.sqrt(self.var + 1e-8), -10, 10)


class EpisodeTracker:
    """Adds up rewards per environment and remembers (step, return) of every finished episode."""

    def __init__(self, n_envs):
        self.running = np.zeros(n_envs)
        self.finished = []

    def update(self, rewards, dones, step):
        self.running += rewards
        for i in np.nonzero(dones)[0]:
            self.finished.append((step, self.running[i]))
            self.running[i] = 0

    def mean_of_last(self, n):
        recent = [ret for _, ret in self.finished[-n:]]
        return float(np.mean(recent)) if recent else float("nan")


def clone_state(state):
    return {key: value.clone() for key, value in state.items()}


def collect_rollout(agent, envs, obs, done, state, tracker, reward_fn, global_step):
    """Play num_steps steps in every environment with the current policy and store everything."""
    T, N = args.num_steps, args.num_envs
    action_shape, action_dtype = ((T, N, envs.n_actions), torch.float32) if CONTINUOUS else ((T, N), torch.long)
    buf = {"obs": torch.zeros((T, N) + tuple(envs.obs_shape), device=DEVICE),
           "actions": torch.zeros(action_shape, device=DEVICE, dtype=action_dtype),
           "logp": torch.zeros(T, N, device=DEVICE),
           "rewards": torch.zeros(T, N, device=DEVICE),
           "dones": torch.zeros(T, N, device=DEVICE),        # dones[t] = 1: step t is the first of a new episode
           "values": torch.zeros(T, N, device=DEVICE)}
    for t in range(T):
        buf["obs"][t], buf["dones"][t] = obs, done
        dist, value, state = agent.step(obs, state, done)
        action = dist.sample()
        buf["actions"][t], buf["values"][t] = action, value
        buf["logp"][t] = dist.log_prob(action).sum(-1) if CONTINUOUS else dist.log_prob(action)

        next_obs, reward, next_done = envs.step(action.cpu().numpy())
        tracker.update(reward, next_done, global_step)
        buf["rewards"][t] = torch.tensor(reward_fn(reward, next_done), device=DEVICE)
        obs, done = torch.tensor(next_obs, device=DEVICE), torch.tensor(next_done, device=DEVICE)
        global_step += N
    return buf, obs, done, state, global_step


def compute_advantages(buf, next_value, next_done, gamma, gae_lambda):
    """Generalized Advantage Estimation, from the last step backwards."""
    T = args.num_steps
    advantages = torch.zeros_like(buf["rewards"])
    running = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            not_done, value_after = 1 - next_done, next_value
        else:
            not_done, value_after = 1 - buf["dones"][t + 1], buf["values"][t + 1]
        td_error = buf["rewards"][t] + gamma * value_after * not_done - buf["values"][t]
        running = td_error + gamma * gae_lambda * not_done * running
        advantages[t] = running
    returns = advantages + buf["values"]
    return advantages, returns


def ppo_update(agent, optimizer, buf, advantages, returns, start_state, clip_coef, ent_coef):
    """Several epochs of clipped PPO. Minibatches are groups of whole environments, so every
    minibatch is re-run as a full sequence from the memory it started with (backprop through time)."""
    N = args.num_envs
    envs_per_batch = N // args.minibatches
    for _ in range(args.epochs):
        order = torch.randperm(N)
        for b in range(args.minibatches):
            idx = order[b * envs_per_batch:(b + 1) * envs_per_batch].to(DEVICE)
            batch_state = {key: (value[:, idx] if key == "S" else value[idx]) for key, value in start_state.items()}
            if hasattr(agent.core, "rates"):
                agent.core.rates.clear()
            dist, values, _ = agent.seq(buf["obs"][:, idx], batch_state, buf["dones"][:, idx])

            logp, entropy = dist.log_prob(buf["actions"][:, idx]), dist.entropy()
            if CONTINUOUS:                                   # independent action dimensions
                logp, entropy = logp.sum(-1), entropy.sum(-1)
            adv = advantages[:, idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = (logp - buf["logp"][:, idx]).exp()
            policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
            value_loss = 0.5 * ((values - returns[:, idx]) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy.mean()
            if args.core == "dendra":                        # homeostasis: keep firing rates near 15%
                rates = torch.stack(agent.core.rates).mean(0)
                loss = loss + ((rates - 0.15) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


# ============================================================================ 5. Main loop
def main():
    N, T = args.num_envs, args.num_steps
    # The memoryless agent gets the last 4 frames instead of a memory (MinAtar frames already show motion)
    frame_stack = 4 if (args.core == "mlp" and ENV_FAMILY in ("atari", "pendulum-pomdp", "mujoco")) else 1
    envs = make_vec_env(frame_stack)
    agent = Agent(envs.obs_shape, envs.n_actions).to(DEVICE)
    n_params = sum(p.numel() for p in agent.parameters())

    # Hyperparameters: the usual PPO values for Atari (discrete) and MuJoCo (continuous)
    lr = args.lr or (3e-4 if CONTINUOUS else 2.5e-4)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    gamma, gae_lambda = 0.99, 0.95
    clip_coef = 0.2 if CONTINUOUS else 0.1
    ent_coef = 0.0 if CONTINUOUS else 0.01
    args.epochs = args.epochs or (10 if CONTINUOUS else 4)
    n_updates = args.total // (N * T)
    if CONTINUOUS:
        reward_fn = RewardNormalizer(N, gamma)
    elif ENV_FAMILY == "atari":
        reward_fn = lambda reward, done: np.sign(reward)    # reward clipping to -1/0/+1
    else:
        reward_fn = lambda reward, done: reward
    print(f"{args.env} | core {args.core} | {n_params / 1e3:.0f}K params | "
          f"{'continuous' if CONTINUOUS else 'discrete'} ({envs.n_actions}) | {n_updates} updates", flush=True)

    obs = torch.tensor(envs.reset(), device=DEVICE)
    done = torch.zeros(N, device=DEVICE)
    state = agent.core.init_state(N)
    tracker = EpisodeTracker(N)
    curve, start_time, global_step = [], time.time(), 0
    for update in range(1, n_updates + 1):
        if args.max_minutes and time.time() - start_time > args.max_minutes * 60:
            print(f"[{args.core}] time limit of {args.max_minutes} min reached, stopping at step {global_step}", flush=True)
            break
        for group in optimizer.param_groups:                  # learning rate decays linearly to 0
            group["lr"] = lr * (1 - (update - 1) / n_updates)

        start_state = clone_state(state)                       # memory at the start of this segment
        with torch.no_grad():
            buf, obs, done, state, global_step = collect_rollout(agent, envs, obs, done, state, tracker, reward_fn, global_step)
            if hasattr(agent.core, "rates"):
                agent.core.rates.clear()
            _, next_value, _ = agent.step(obs, clone_state(state), done)
            advantages, returns = compute_advantages(buf, next_value, done, gamma, gae_lambda)
        ppo_update(agent, optimizer, buf, advantages, returns, start_state, clip_coef, ent_coef)

        if update % max(1, n_updates // 40) == 0 or update == n_updates:
            elapsed = time.time() - start_time
            avg = tracker.mean_of_last(50)
            curve.append({"step": global_step, "return": avg, "time": elapsed})
            print(f"[{args.core}] step {global_step:8d} | mean return (last 50 episodes) {avg:8.2f} | "
                  f"{global_step / elapsed:5.0f} steps/s | {elapsed / 60:5.1f} min", flush=True)

    last_100 = [ret for _, ret in tracker.finished[-100:]]
    results = {"env": args.env, "core": args.core, "seed": args.seed, "total": global_step, "params": n_params,
               "final_return_mean": float(np.mean(last_100)) if last_100 else None, "curve": curve,
               "episodes": [(int(step), float(ret)) for step, ret in tracker.finished],
               "time_s": time.time() - start_time}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[{args.core}] DONE: mean return (last 100 episodes) {results['final_return_mean']}, "
          f"{results['time_s'] / 60:.1f} min", flush=True)
    return agent


if __name__ == "__main__":
    agent = main()
