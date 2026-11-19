"""
Recurrent PPO for both DISCRETE actions (Atari, MinAtar) and CONTINUOUS actions (Pendulum, MuJoCo).
Swappable core: DENDRA (Hebbian synapses + spiking neurons + neuromodulators), LSTM, or a memoryless MLP/CNN.

Environments (--env):
  atari:Breakout          real Atari 2600 (ALE), 84x84 grayscale, frame skip 4
  minatar:breakout        miniature 10x10 Atari (MinAtar), fast on CPU
  pendulum-pomdp          Pendulum with the angular velocity HIDDEN -> must be inferred from memory (continuous actions)
  mujoco:HalfCheetah-v5   full MuJoCo; add the suffix "-pomdp" to keep positions only (velocities hidden)

Example:
  python experiments/rl/ppo_recurrent.py --env minatar:breakout --core dendra --total 300000 --out runs/ppo.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dendra.model import CorticalRegion, Neuromodulators

p = argparse.ArgumentParser()
p.add_argument("--env", default="minatar:breakout")
p.add_argument("--core", default="dendra", choices=["dendra", "lstm", "mlp"])
p.add_argument("--total", type=int, default=300_000, help="total environment steps")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--num_steps", type=int, default=128)
p.add_argument("--minibatches", type=int, default=4)
p.add_argument("--epochs", type=int, default=None, help="default: 4 (discrete), 10 (continuous)")
p.add_argument("--lr", type=float, default=None)
p.add_argument("--d", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_minutes", type=float, default=None, help="stop early (results are still saved) after this many minutes")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out", default="runs/ppo.json")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(int(os.environ.get("NT", "1")))
dev = torch.device(a.device)
KIND = a.env.split(":")[0]
CONT = KIND in ("pendulum-pomdp", "mujoco")


# ================================================================== environments
class MinAtarVec:
    """Vectorized MinAtar, auto-resets at episode end. Observations [N, C, 10, 10]."""

    def __init__(self, game, n, seed):
        from minatar import Environment
        self.envs = [Environment(game) for i in range(n)]
        for i, e in enumerate(self.envs):
            e.random = np.random.RandomState(seed * 1000 + i)   # independent random stream per environment
            e.reset()
        self.n_actions = self.envs[0].num_actions()
        self.obs_shape = (self.envs[0].state_shape()[2], 10, 10)

    def obs(self):
        return np.stack([e.state().transpose(2, 0, 1) for e in self.envs]).astype(np.float32)

    def reset(self):
        return self.obs()

    def step(self, actions):
        rew, done = np.zeros(len(self.envs), np.float32), np.zeros(len(self.envs), np.float32)
        for i, (e, act) in enumerate(zip(self.envs, actions)):
            r, d = e.act(int(act))
            rew[i], done[i] = r, float(d)
            if d:
                e.reset()
        return self.obs(), rew, done


class GymVec:
    """Gymnasium SyncVectorEnv that resets within the terminating step (SAME_STEP)."""

    def __init__(self, thunk, n, seed):
        self.v = gym.vector.SyncVectorEnv([thunk(i) for i in range(n)],
                                          autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.seed = seed
        sp = self.v.single_action_space
        self.n_actions = sp.n if hasattr(sp, "n") else sp.shape[0]
        self.obs_shape = self.v.single_observation_space.shape
        if CONT:
            self.low, self.high = sp.low, sp.high

    def reset(self):
        return self.v.reset(seed=self.seed)[0].astype(np.float32)

    def step(self, actions):
        if CONT:
            actions = np.clip(actions, self.low, self.high)
        o, r, term, trunc, _ = self.v.step(actions)
        return o.astype(np.float32), r.astype(np.float32), (term | trunc).astype(np.float32)


def make_vec(stack):
    if KIND == "minatar":
        return MinAtarVec(a.env.split(":")[1], a.num_envs, a.seed)
    if KIND == "atari":
        import ale_py
        gym.register_envs(ale_py)
        game = a.env.split(":")[1]

        def thunk(i):
            def f():
                e = gym.make(f"ALE/{game}-v5", frameskip=1)
                e = gym.wrappers.AtariPreprocessing(e, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
                return gym.wrappers.FrameStackObservation(e, stack)
            return f
        return GymVec(thunk, a.num_envs, a.seed)
    if KIND == "pendulum-pomdp":
        def thunk(i):
            def f():
                e = gym.make("Pendulum-v1")
                e = gym.wrappers.TransformObservation(e, lambda o: o[:2].astype(np.float32),
                                                      gym.spaces.Box(-1.0, 1.0, (2,), np.float32))  # drop angular velocity
                if stack > 1:
                    e = gym.wrappers.FrameStackObservation(e, stack)
                    e = gym.wrappers.FlattenObservation(e)
                return e
            return f
        return GymVec(thunk, a.num_envs, a.seed)
    if KIND == "mujoco":
        name = a.env.split(":")[1]
        pomdp = name.endswith("-pomdp")
        name = name.replace("-pomdp", "")

        def thunk(i):
            def f():
                e = gym.make(name)
                if pomdp:   # keep joint positions only (first nq - 1 values), hide all velocities
                    k = e.unwrapped.model.nq - 1
                    e = gym.wrappers.TransformObservation(e, lambda o: o[:k].astype(np.float32),
                                                          gym.spaces.Box(-np.inf, np.inf, (k,), np.float32))
                if stack > 1:
                    e = gym.wrappers.FrameStackObservation(e, stack)
                    e = gym.wrappers.FlattenObservation(e)
                return e
            return f
        return GymVec(thunk, a.num_envs, a.seed)
    raise ValueError(a.env)


# ================================================================== networks
def layer(m, std=math.sqrt(2)):
    nn.init.orthogonal_(m.weight, std); nn.init.zeros_(m.bias)
    return m


class Encoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        if len(shape) == 3 and shape[-1] == 84:           # Atari
            self.net = nn.Sequential(layer(nn.Conv2d(shape[0], 32, 8, 4)), nn.ReLU(), layer(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
                                     layer(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(), nn.Flatten(), layer(nn.Linear(3136, 512)), nn.ReLU())
            self.dim, self.scale = 512, 1 / 255.0
        elif len(shape) == 3:                             # MinAtar 10x10
            self.net = nn.Sequential(layer(nn.Conv2d(shape[0], 16, 3, 1)), nn.ReLU(), nn.Flatten(),
                                     layer(nn.Linear(16 * 8 * 8, 128)), nn.ReLU())
            self.dim, self.scale = 128, 1.0
        else:                                             # vector
            self.net = nn.Sequential(layer(nn.Linear(int(np.prod(shape)), 64)), nn.Tanh(), layer(nn.Linear(64, 64)), nn.Tanh())
            self.dim, self.scale = 64, 1.0

    def forward(self, x):
        return self.net(x * self.scale)


class MLPCore(nn.Module):
    def __init__(self, F_, d):
        super().__init__(); self.fc = layer(nn.Linear(F_, d))

    def init_state(self, n):
        return {}

    def step(self, x, st, done):
        return torch.tanh(self.fc(x)) if CONT else F.relu(self.fc(x)), st


class LSTMCore(nn.Module):
    def __init__(self, F_, d):
        super().__init__(); self.cell = nn.LSTMCell(F_, d)

    def init_state(self, n):
        z = torch.zeros(n, self.cell.hidden_size, device=dev)
        return {"h": z, "c": z.clone()}

    def step(self, x, st, done):
        keep = (1 - done)[:, None]
        h, c = self.cell(x, (st["h"] * keep, st["c"] * keep))
        return h, {"h": h, "c": c}


class DendraCore(nn.Module):
    """DENDRA core run step by step: Hebbian synapses S, spiking dendritic E/I neurons, neuromodulators.
    The state is cleared when a new episode starts (done = 1)."""

    def __init__(self, F_, d, regions=2, heads=4):
        super().__init__()
        self.inp = layer(nn.Linear(F_, d), 1.0)
        self.regions = nn.ModuleList([CorticalRegion(d, heads, int(1.5 * d), 4, 4) for _ in range(regions)])
        self.nm = Neuromodulators(d, regions, heads)
        self.L, self.h, self.d = regions, heads, d
        self.rates = []

    def init_state(self, n):
        dh = self.d // self.h
        return {"S": torch.zeros(self.L, n, self.h, dh, dh, device=dev), "m": torch.zeros(n, self.d, device=dev),
                "t": torch.zeros(n, device=dev)}

    def step(self, x, st, done):
        keep = 1 - done
        S = st["S"] * keep[None, :, None, None, None]
        m, t = st["m"] * keep[:, None], st["t"] * keep
        x = self.inp(x)
        dec = self.nm.decay
        m = dec * m + x
        norm = (1 - dec ** (t + 1)) / (1 - dec)
        z = 2 * torch.sigmoid(self.nm.proj(self.nm.norm(m / norm[:, None]))).view(-1, self.L, 1 + self.h)
        gain, ach = z[..., :1], z[..., 1:]
        h, newS = x[:, None], []
        for l, reg in enumerate(self.regions):
            h, Sl = reg.step(h, gain[:, None, l], ach[:, None, l], None, S[l])
            newS.append(Sl)
        self.rates.append(torch.stack([r.l5.last_rate for r in self.regions]))
        return h[:, 0], {"S": torch.stack(newS), "m": m, "t": t + 1}

    def seq(self, x, st, done):
        """A whole segment of T steps at once (parallel over time) - same result as calling step() T times.
        x [T, N, F]; done [T, N] (1 = this step starts a new episode -> clear the state before it)."""
        Tn, N = done.shape
        x = self.inp(x)                                                     # [T, N, d]
        dec = self.nm.decay
        m, t, ms, ts = st["m"], st["t"], [], []
        for i in range(Tn):                                                  # neuromodulator baseline level (cheap)
            keep = 1 - done[i]
            m = dec * m * keep[:, None] + x[i]; t = t * keep
            ms.append(m); ts.append(t); t = t + 1
        M, Tt = torch.stack(ms), torch.stack(ts)
        norm = (1 - dec ** (Tt + 1)) / (1 - dec)
        z = 2 * torch.sigmoid(self.nm.proj(self.nm.norm(M / norm[..., None]))).view(Tn, N, self.L, 1 + self.h)
        gain, ach = z[..., :1].transpose(0, 1), z[..., 1:].transpose(0, 1)  # [N, T, L, .]
        seg = done.transpose(0, 1).cumsum(1)                                # [N, T] number of resets up to step t
        same = (seg[:, :, None] == seg[:, None, :])                          # same episode
        first = (seg == 0)                                                   # no reset yet -> still sees the old state
        idx = torch.arange(Tn, device=x.device)
        dist = (idx[:, None] - idx[None, :]).float()
        h, newS, rates = x.transpose(0, 1), [], []
        for l, reg in enumerate(self.regions):
            a23 = reg.l23
            xin = reg.l4a(h)
            q, k, v, w = a23._proj(xin, ach[:, :, l])                       # [N,T,h,dh], w [N,T,h]
            loglam = torch.log(a23.lam())                                    # [h]
            D = torch.where(dist >= 0, torch.exp(loglam[:, None, None] * dist.clamp(min=0)), torch.zeros((), device=x.device))
            D = D[None] * same[:, None]                                      # [N,h,T,T]
            A = torch.einsum("bihd,bjhd->bhij", q, k) * D * w.permute(0, 2, 1)[:, :, None, :]
            o = torch.einsum("bhij,bjhd->bihd", A, v)
            S0 = st["S"][l]                                                  # [N,h,dh,dh]
            dec_in = torch.exp(loglam[None, :] * (idx[:, None] + 1).float())                      # [T,h]
            o = o + torch.einsum("bhde,bihe->bihd", S0, q) * (dec_in[None, :, :, None] * first[:, :, None, None])
            last = seg[:, -1:]
            dec_out = torch.exp(loglam[None, :] * (Tn - 1 - idx)[:, None].float())                # [T,h]
            wv = v * (w * dec_out[None] * (seg == last)[:, :, None])[..., None]
            S_end = (torch.exp(loglam * Tn)[None, :, None, None] * S0 * (seg[:, -1] == 0)[:, None, None, None]
                     + torch.einsum("bjhd,bjhe->bhde", wv, k))
            h = h + a23._finish(xin, o)
            h = h + reg.l5(reg.l4b(h), gain[:, :, l])
            newS.append(S_end); rates.append(reg.l5.last_rate)
        self.rates.append(torch.stack(rates))
        return h.transpose(0, 1), {"S": torch.stack(newS), "m": M[-1], "t": Tt[-1] + 1}


class Agent(nn.Module):
    def __init__(self, obs_shape, n_act):
        super().__init__()
        self.enc = Encoder(obs_shape)
        self.core = {"dendra": DendraCore, "lstm": LSTMCore, "mlp": MLPCore}[a.core](self.enc.dim, a.d)
        self.pi = layer(nn.Linear(a.d, n_act), 0.01)
        self.v = layer(nn.Linear(a.d, 1), 1.0)
        if CONT:
            self.log_std = nn.Parameter(torch.zeros(n_act))

    def dist(self, h):
        if CONT:
            return torch.distributions.Normal(self.pi(h), self.log_std.exp())
        return torch.distributions.Categorical(logits=self.pi(h))

    def step(self, obs, st, done):
        h, st = self.core.step(self.enc(obs), st, done)
        return self.dist(h), self.v(h)[:, 0], st

    def seq(self, obs, st, done):
        """A whole segment [T, N, ...]: DENDRA runs in parallel over time; LSTM/MLP loop step by step."""
        Tn, N = done.shape
        f = self.enc(obs.flatten(0, 1)).view(Tn, N, -1)
        if hasattr(self.core, "seq"):
            h, st = self.core.seq(f, st, done)
        else:
            hs = []
            for t in range(Tn):
                ht, st = self.core.step(f[t], st, done[t]); hs.append(ht)
            h = torch.stack(hs)
        return self.dist(h), self.v(h)[..., 0], st


# ================================================================== PPO
class RunningStd:   # reward normalization by the std of the discounted return (continuous control)
    def __init__(self, n, gamma):
        self.ret, self.g, self.cnt, self.mean, self.var = np.zeros(n), gamma, 1e-4, 0.0, 1.0

    def __call__(self, r, done):
        self.ret = self.ret * self.g + r
        b_mean, b_var, b_cnt = self.ret.mean(), self.ret.var(), len(r)
        delta, tot = b_mean - self.mean, self.cnt + b_cnt
        self.mean += delta * b_cnt / tot
        self.var = (self.var * self.cnt + b_var * b_cnt + delta ** 2 * self.cnt * b_cnt / tot) / tot
        self.cnt = tot
        self.ret[done > 0] = 0
        return np.clip(r / math.sqrt(self.var + 1e-8), -10, 10)


stack = 4 if (a.core == "mlp" and KIND in ("atari", "pendulum-pomdp", "mujoco")) else 1   # the memoryless MLP sees the last 4 frames
envs = make_vec(stack)
agent = Agent(envs.obs_shape, envs.n_actions).to(dev)
lr = a.lr or (3e-4 if CONT else 2.5e-4)
opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
gamma, lam_gae = 0.99, 0.95
clip = 0.2 if CONT else 0.1
ent_coef = 0.0 if CONT else 0.01
N, Tn = a.num_envs, a.num_steps
a.epochs = a.epochs or (10 if CONT else 4)
n_updates = a.total // (N * Tn)
rnorm = RunningStd(N, gamma) if CONT else None
print(f"{a.env} | core {a.core} | {sum(p.numel() for p in agent.parameters())/1e3:.0f}K params | "
      f"{'continuous' if CONT else 'discrete'} ({envs.n_actions}) | {n_updates} updates", flush=True)

obs = torch.tensor(envs.reset(), device=dev)
done = torch.zeros(N, device=dev)
st = agent.core.init_state(N)
ep_ret, ep_len = np.zeros(N), np.zeros(N)
finished, curve, t0, gstep = [], [], time.time(), 0
for upd in range(1, n_updates + 1):
    if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
        print(f"[{a.core}] time limit of {a.max_minutes} min reached, stopping at step {gstep}", flush=True)
        break
    for g_ in opt.param_groups:
        g_["lr"] = lr * (1 - (upd - 1) / n_updates)
    init_st = {k: v.clone() for k, v in st.items()}
    B_obs = torch.zeros((Tn, N) + tuple(envs.obs_shape), device=dev)
    B_act = torch.zeros((Tn, N, envs.n_actions) if CONT else (Tn, N), device=dev, dtype=torch.float32 if CONT else torch.long)
    B_logp, B_rew, B_done, B_val = (torch.zeros(Tn, N, device=dev) for _ in range(4))
    with torch.no_grad():
        for t in range(Tn):
            B_obs[t], B_done[t] = obs, done
            dist, val, st = agent.step(obs, st, done)
            act = dist.sample()
            B_act[t], B_val[t] = act, val
            B_logp[t] = dist.log_prob(act).sum(-1) if CONT else dist.log_prob(act)
            o2, r, d = envs.step(act.cpu().numpy())
            ep_ret += r; ep_len += 1
            for i in np.nonzero(d)[0]:
                finished.append((gstep, ep_ret[i])); ep_ret[i] = 0; ep_len[i] = 0
            r_train = rnorm(r, d) if CONT else np.sign(r) if KIND == "atari" else r
            B_rew[t] = torch.tensor(r_train, device=dev)
            obs, done = torch.tensor(o2, device=dev), torch.tensor(d, device=dev)
            gstep += N
        if hasattr(agent.core, "rates"):
            agent.core.rates.clear()
        _, next_v, _ = agent.step(obs, {k: v.clone() for k, v in st.items()}, done)
        adv = torch.zeros(Tn, N, device=dev); last = torch.zeros(N, device=dev)
        for t in reversed(range(Tn)):
            nonterm = 1 - (done if t == Tn - 1 else B_done[t + 1])
            nv = next_v if t == Tn - 1 else B_val[t + 1]
            delta = B_rew[t] + gamma * nv * nonterm - B_val[t]
            last = delta + gamma * lam_gae * nonterm * last
            adv[t] = last
        ret = adv + B_val

    env_per_mb = N // a.minibatches
    for ep in range(a.epochs):
        perm = torch.randperm(N)
        for k in range(a.minibatches):
            idx = perm[k * env_per_mb:(k + 1) * env_per_mb].to(dev)
            s_ = {kk: (v[:, idx] if kk == "S" else v[idx]) for kk, v in init_st.items()}
            if hasattr(agent.core, "rates"):
                agent.core.rates.clear()
            dist, val, _ = agent.seq(B_obs[:, idx], s_, B_done[:, idx])        # re-run the whole segment (BPTT)
            logp = dist.log_prob(B_act[:, idx]); ent = dist.entropy()
            if CONT:
                logp, ent = logp.sum(-1), ent.sum(-1)
            mb_adv = adv[:, idx]; mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            ratio = (logp - B_logp[:, idx]).exp()
            pg = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - clip, 1 + clip)).mean()
            vloss = 0.5 * ((val - ret[:, idx]) ** 2).mean()
            loss = pg + 0.5 * vloss - ent_coef * ent.mean()
            if a.core == "dendra":
                rates = torch.stack(agent.core.rates).mean(0)
                loss = loss + ((rates - 0.15) ** 2).sum()   # homeostasis
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()

    if upd % max(1, n_updates // 40) == 0 or upd == n_updates:
        recent = [r for _, r in finished[-50:]]
        avg = float(np.mean(recent)) if recent else float("nan")
        curve.append({"step": gstep, "return": avg, "time": time.time() - t0})
        sps = gstep / (time.time() - t0)
        print(f"[{a.core}] step {gstep:8d} | mean return (last 50 episodes) {avg:8.2f} | {sps:5.0f} steps/s | "
              f"{(time.time()-t0)/60:5.1f} min", flush=True)

final = [r for _, r in finished[-100:]]
res = {"env": a.env, "core": a.core, "seed": a.seed, "total": gstep, "params": sum(p.numel() for p in agent.parameters()),
       "final_return_mean": float(np.mean(final)) if final else None, "curve": curve,
       "episodes": [(int(s), float(r)) for s, r in finished], "time_s": time.time() - t0}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"))
print(f"[{a.core}] DONE: mean return (last 100 episodes) {res['final_return_mean']}, {res['time_s']/60:.1f} min", flush=True)
