"""
Meta-RL on Bernoulli bandits: comparing DENDRA, DENDRA + dopamine, GPT and LSTM.

Every episode is a new machine whose arm win probabilities p_i ~ U(0, 1) are hidden from the agent.
The agent pulls T times; its input at each step = (previous action, previous reward, t/T).
To do well the network must "learn in context": try the arms, remember the outcomes, then exploit the best arm
(Wang et al., 2018; Duan et al., 2016). Network weights do NOT change within an episode; all within-episode
learning happens in the network state: Hebbian synapses (DENDRA), memory cells (LSTM), context (GPT).

Learning algorithm: A2C (actor-critic), gamma = 0.9, decaying entropy bonus.
Evaluation: cumulative regret = sum_t (p_max - p_{a_t}) over 2,000 fixed episodes; compared with Thompson Sampling,
UCB1 and random choice.

Run: python experiments/rl/bandit.py --model dendra --updates 1500 --out runs/rl_dendra.json
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root
from dendra.model import CorticalRegion, Neuromodulators
from experiments.baselines.gpt import GPTBlock

p = argparse.ArgumentParser()
p.add_argument("--model", default="dendra", choices=["dendra", "dendra_da", "gpt", "lstm"])
p.add_argument("--arms", type=int, default=2)
p.add_argument("--T", type=int, default=100)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--updates", type=int, default=1500)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--gamma", type=float, default=0.9)
p.add_argument("--d", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_n", type=int, default=2000)
p.add_argument("--out", default="runs/rl.json")
a = p.parse_args()
torch.manual_seed(a.seed)
torch.set_num_threads(int(os.environ.get("NT", "1")))
A, T = a.arms, a.T
OBS = A + 2


# ------------------------------------------------------------------ agents
class Heads(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pi, self.v = nn.Linear(d, A), nn.Linear(d, 1)

    def forward(self, h):
        h = self.norm(h)
        return self.pi(h), self.v(h).squeeze(-1)


class DendraAgent(nn.Module):
    """DENDRA with continuous inputs. dopamine=True: a reward-prediction-error (RPE) signal opens/closes the write
    gate of the Hebbian synapses ("learn when surprised"), on top of the existing acetylcholine gate."""

    def __init__(self, d, regions=2, heads=4, n_e=96, dopamine=False):
        super().__init__()
        self.inp = nn.Linear(OBS, d)
        self.regions = nn.ModuleList([CorticalRegion(d, heads, n_e, 4, 4) for _ in range(regions)])
        self.nm = Neuromodulators(d, regions, heads)
        self.dopamine = dopamine
        if dopamine:
            self.da_w = nn.Parameter(torch.zeros(regions, heads))   # starts with no effect
            self.da_b = nn.Parameter(torch.zeros(regions, heads))
        self.out = Heads(d)

    def rpe(self, obs):
        # causal RPE computed from the inputs: reward just received - expected level (moving average of earlier rewards)
        r = obs[..., A]
        base, avg = torch.zeros_like(r[:, 0]), []
        for t in range(r.shape[1]):
            avg.append(base)
            base = 0.8 * base + 0.2 * r[:, t]
        return r - torch.stack(avg, 1)                                     # [B, T]

    def forward(self, obs):
        x = self.inp(obs)
        gain, ach = self.nm(x)                                             # [B,T,L,1], [B,T,L,h]
        if self.dopamine:
            delta = self.rpe(obs)[..., None, None]                         # [B,T,1,1]
            ach = ach * 2 * torch.sigmoid(self.da_w * delta + self.da_b)   # dopamine gate
        h = x
        for l, reg in enumerate(self.regions):
            h = reg(h, gain[:, :, l], ach[:, :, l])
        return self.out(h)

    def spike_rates(self):
        return [r.l5.last_rate for r in self.regions]

    def init_state(self, B):
        h = self.regions[0].l23.h; dh = self.inp.out_features // h
        return {"S": [torch.zeros(B, h, dh, dh) for _ in self.regions], "m": torch.zeros(B, 1, self.inp.out_features),
                "t": 0, "base": torch.zeros(B)}

    def step(self, obs_t, st):
        """One decision step with a constant-size state (synapses S, neuromodulator baseline, expected reward)."""
        x = self.inp(obs_t)[:, None]
        (gain, ach), st["m"] = self.nm.step(x, st["m"], st["t"]); st["t"] += 1
        if self.dopamine:
            r = obs_t[:, A]; delta = r - st["base"]; st["base"] = 0.8 * st["base"] + 0.2 * r
            ach = ach * 2 * torch.sigmoid(self.da_w * delta[:, None, None, None] + self.da_b)
        h = x
        for l, reg in enumerate(self.regions):
            h, st["S"][l] = reg.step(h, gain[:, :, l], ach[:, :, l], None, st["S"][l])
        lg, v = self.out(h[:, 0])
        return lg, st


class GPTAgent(nn.Module):
    """Causal Transformer (pre-LN). During play it uses a KV cache: each step only computes the new token."""

    def __init__(self, d, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(OBS, d)
        self.pos = nn.Embedding(1024, d)
        self.blocks = nn.ModuleList([GPTBlock(d, heads) for _ in range(layers)])
        self.out = Heads(d)

    def forward(self, obs):
        x = self.inp(obs) + self.pos(torch.arange(obs.shape[1]).clamp(max=1023))
        for b in self.blocks:
            x = b(x)
        return self.out(x)

    def init_state(self, B):
        return {"kv": [None] * len(self.blocks), "t": 0}

    def step(self, obs_t, st):
        x = self.inp(obs_t)[:, None] + self.pos(torch.tensor([min(st["t"], 1023)]))
        for i, b in enumerate(self.blocks):
            B, _, D = x.shape
            q, k, v = b.qkv(b.ln1(x)).view(B, 1, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
            if st["kv"][i] is not None:
                k = torch.cat([st["kv"][i][0], k], 2); v = torch.cat([st["kv"][i][1], v], 2)
            st["kv"][i] = (k, v)
            att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, 1, D)
            x = x + b.proj(att)
            x = x + b.mlp(b.ln2(x))
        st["t"] += 1
        lg, v_ = self.out(x[:, 0])
        return lg, st


class LSTMAgent(nn.Module):
    def __init__(self, d, hidden=160):
        super().__init__()
        self.inp = nn.Linear(OBS, d)
        self.lstm = nn.LSTM(d, hidden, batch_first=True)      # large hidden size so the parameter count matches the others
        self.out = Heads(hidden)

    def forward(self, obs):
        return self.out(self.lstm(self.inp(obs))[0])

    def init_state(self, B):
        return None

    def step(self, obs_t, st):
        y, st = self.lstm(self.inp(obs_t)[:, None], st)
        lg, v = self.out(y[:, 0])
        return lg, st


def make():
    if a.model == "dendra":
        return DendraAgent(a.d)
    if a.model == "dendra_da":
        return DendraAgent(a.d, dopamine=True)
    if a.model == "gpt":
        return GPTAgent(a.d)
    return LSTMAgent(a.d)


# ------------------------------------------------------------------ environment
@torch.no_grad()
def rollout(model, probs, greedy=False, gen=None, T_ep=None):
    """Play one episode for the whole batch (no gradient)."""
    T_ep = T_ep or T
    B = probs.shape[0]
    obs = torch.zeros(B, T_ep, OBS)
    acts = torch.zeros(B, T_ep, dtype=torch.long)
    rews = torch.zeros(B, T_ep)
    for t in range(T_ep):
        if t > 0:
            obs[torch.arange(B), t, acts[:, t - 1]] = 1.0
            obs[:, t, A] = rews[:, t - 1]
        obs[:, t, A + 1] = t / T
        if hasattr(model, "step"):                       # DENDRA, LSTM: constant-size state; GPT: KV cache
            if t == 0: st = model.init_state(B)
            lg, st = model.step(obs[:, t], st)
        else:
            logits, _ = model(obs[:, :t + 1])
            lg = logits[:, -1]
        at = lg.argmax(-1) if greedy else torch.multinomial(lg.softmax(-1), 1, generator=gen)[:, 0]
        acts[:, t] = at
        rews[:, t] = (torch.rand(B, generator=gen) < probs[torch.arange(B), at]).float()
    return obs, acts, rews


def regret_curve(acts, probs):
    best = probs.max(-1).values[:, None]
    chosen = probs.gather(1, acts)
    return (best - chosen).cumsum(1).mean(0)            # expected cumulative regret over t


# ------------------------------------------------------------------ A2C training
model = make()
n_params = sum(p.numel() for p in model.parameters())
print(f"{a.model}: {n_params/1e3:.1f}K params", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
disc = a.gamma ** torch.arange(T).float()
log, t0 = [], time.time()
for u in range(a.updates + 1):
    probs = torch.rand(a.batch, A)
    model.eval()
    obs, acts, rews = rollout(model, probs)
    model.train()
    logits, values = model(obs)                          # one parallel pass with gradients
    ret = torch.zeros_like(rews); run = torch.zeros(a.batch)
    for t in reversed(range(T)):
        run = rews[:, t] + a.gamma * run
        ret[:, t] = run
    adv = (ret - values).detach()
    logp = F.log_softmax(logits, -1)
    ent = -(logp.exp() * logp).sum(-1).mean()
    beta = 0.05 * max(0.0, 1 - u / (0.8 * a.updates)) + 0.005
    loss = -(logp.gather(-1, acts[..., None])[..., 0] * adv).mean() + 0.5 * ((ret - values) ** 2).mean() - beta * ent
    if a.model.startswith("dendra"):
        rates = torch.stack(model.spike_rates())
        loss = loss + ((rates - 0.15) ** 2).sum()        # homeostasis
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if u % 50 == 0:
        reg = regret_curve(acts, probs)[-1].item()
        log.append({"update": u, "regret": reg, "reward": rews.mean().item(), "time": time.time() - t0})
        print(f"[{a.model}] update {u:5d} | regret (sampled) {reg:6.2f} | mean reward {rews.mean():.3f} | "
              f"{time.time()-t0:5.0f}s", flush=True)
train_time = time.time() - t0


# ------------------------------------------------------------------ evaluation + classic baselines
def baselines(probs, gen, T_ep):
    B = probs.shape[0]
    out = {}
    # random
    acts = torch.randint(0, A, (B, T_ep), generator=gen); out["random"] = regret_curve(acts, probs)
    # Thompson Sampling (Beta-Bernoulli)
    al, be = torch.ones(B, A), torch.ones(B, A); acts = torch.zeros(B, T_ep, dtype=torch.long)
    for t in range(T_ep):
        smp = torch.distributions.Beta(al, be).sample()
        at = smp.argmax(-1); acts[:, t] = at
        r = (torch.rand(B, generator=gen) < probs[torch.arange(B), at]).float()
        al[torch.arange(B), at] += r; be[torch.arange(B), at] += 1 - r
    out["thompson"] = regret_curve(acts, probs)
    # UCB1
    n, s = torch.zeros(B, A), torch.zeros(B, A); acts = torch.zeros(B, T_ep, dtype=torch.long)
    for t in range(T_ep):
        if t < A:
            at = torch.full((B,), t)
        else:
            at = (s / n + torch.sqrt(2 * math.log(t) / n)).argmax(-1)
        acts[:, t] = at
        r = (torch.rand(B, generator=gen) < probs[torch.arange(B), at]).float()
        n[torch.arange(B), at] += 1; s[torch.arange(B), at] += r
    out["ucb1"] = regret_curve(acts, probs)
    return out


model.eval()
res = {"model": a.model, "params": n_params, "updates": a.updates, "train_time_s": train_time, "log": log}
for T_ep in (T, 3 * T):                                       # 3T: play longer than during training
    g = torch.Generator().manual_seed(1234)
    N = a.eval_n
    probs = torch.rand(N, A, generator=g)
    curves = []
    for i in range(0, N, 250):
        pb = probs[i:i + 250]
        _, acts, _ = rollout(model, pb, gen=g, T_ep=T_ep)
        curves.append(regret_curve(acts, pb) * pb.shape[0])
    curve = torch.stack(curves).sum(0) / N
    key = "eval" if T_ep == T else "eval_long"
    res[key] = {"T": T_ep, "regret_curve": curve.tolist(), "final_regret": curve[-1].item()}
    g2 = torch.Generator().manual_seed(1234); probs2 = torch.rand(N, A, generator=g2)
    res[key]["baselines"] = {k: v[-1].item() for k, v in baselines(probs2, g2, T_ep).items()}
    res[key]["baseline_curves"] = {k: v.tolist() for k, v in baselines(probs2, torch.Generator().manual_seed(99), T_ep).items()}
    print(f"[{a.model}] EVAL T={T_ep}: regret {curve[-1]:.2f} | "
          + " | ".join(f"{k} {v:.2f}" for k, v in res[key]["baselines"].items()), flush=True)

# speed: time of one decision step when T-1 steps have already been played (batch 1)
with torch.no_grad():
    o = torch.zeros(1, T, OBS)
    if hasattr(model, "step"):
        st = model.init_state(1)
        for t in range(T - 1): _, st = model.step(o[:, t], st)
        import copy
        f = lambda: model.step(o[:, -1], copy.deepcopy(st))
    else:
        f = lambda: model(o)
    f(); t1 = time.perf_counter()
    for _ in range(50): f()
    res["decision_ms"] = (time.perf_counter() - t1) / 50 * 1000
if a.model == "dendra_da":   # learned dopamine gate: 2*sigmoid(w*RPE + b) per region/head
    res["dopamine_w"] = model.da_w.detach().tolist(); res["dopamine_b"] = model.da_b.detach().tolist()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
torch.save(model.state_dict(), a.out.replace(".json", ".pt"))
