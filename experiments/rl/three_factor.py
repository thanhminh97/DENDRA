"""
Can DENDRA learn without backpropagation and without PPO?

The same DENDRA agent (dendra.rl.DendraCore + a policy readout + a value readout) is trained on CartPole in three ways:

  backprop   reference: A2C with backpropagation through time (BPTT) over 32-step segments,
             the example from the README.
  readout    three-factor rule on the OUTPUT synapses only. The DENDRA network itself keeps its random
             initial weights (its Hebbian synapses still work as memory inside an episode).
  eprop      three-factor rule on EVERY synapse (e-prop-like): each region gets the output error through
             fixed random feedback weights. No backpropagation between regions and none through time.
  online_backprop   control for eprop: the same online, one-step schedule and learning rates, but with
             exact gradients (backpropagation through the whole network at each step, still none through time).
             It separates the effect of the learning RULE from the effect of the update SCHEDULE.

Three-factor rule (dopamine-modulated plasticity), applied online at every step:
    dw = lr * delta_t * e_t
    delta_t = r_t + gamma * V(s_t+1) - V(s_t)    "dopamine": reward prediction error, one number per agent,
                                                  broadcast to every synapse
    e_t     = pre-synaptic activity x post-synaptic error, computed at the synapse itself
              output synapses: (onehot(a_t) - pi_t) x h_t  for the policy, h_t for the value
              hidden synapses: post error = B_region (onehot(a_t) - pi_t) + b_region   (B, b fixed and random)

What is still not fully local in "eprop": inside one cortical region (L4 -> L2/3 -> L5 and its 4 spike
ticks) the eligibility is computed with autograd and surrogate gradients. Nothing crosses regions or time.
The neuromodulator weights stay at their initial values (gain = ACh = 1) in both three-factor variants.

Tasks: cartpole (full observation) and cartpole-pomdp (only cart position and pole angle; the velocities
must be inferred from memory).

Run:  python experiments/rl/three_factor.py --method eprop --task cartpole --seed 0 --out runs/tf.json
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dendra.rl import DendraAgent

parser = argparse.ArgumentParser(description="DENDRA trained with backprop or with a three-factor (dopamine) rule")
parser.add_argument("--method", default="eprop", choices=["backprop", "online_backprop", "readout", "eprop"])
parser.add_argument("--task", default="cartpole", choices=["cartpole", "cartpole-pomdp"])
parser.add_argument("--steps", type=int, default=100_000, help="total environment steps")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--lr", type=float, default=None, help="default: 3e-4 (backprop), 1e-3 (three-factor)")
parser.add_argument("--critic_mult", type=float, default=10.0, help="three-factor: value readout learns this much faster")
parser.add_argument("--hidden_mult", type=float, default=1.0, help="eprop / online_backprop: hidden synapses learn at lr * hidden_mult")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="runs/three_factor.json")
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(1)

N = args.num_envs
VISIBLE = [0, 2] if args.task == "cartpole-pomdp" else [0, 1, 2, 3]   # pomdp: cart position, pole angle
SAME_STEP = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
envs = gym.make_vec("CartPole-v1", num_envs=N, vectorization_mode="sync", vector_kwargs=SAME_STEP)
agent = DendraAgent(obs_dim=len(VISIBLE), n_actions=2)
lr = args.lr or (3e-4 if args.method == "backprop" else 1e-3)


def observe(raw):
    return torch.as_tensor(raw[:, VISIBLE], dtype=torch.float32)


class Tracker:
    def __init__(self):
        self.running, self.finished = np.zeros(N), []

    def update(self, reward, ended, step):
        self.running += reward
        for i in np.nonzero(ended)[0]:
            self.finished.append((step, float(self.running[i])))
            self.running[i] = 0


# ============================================================================ reference: backprop (A2C + BPTT)
def train_backprop(tracker, T=32):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    obs, done = observe(envs.reset(seed=args.seed)[0]), torch.zeros(N)
    state, step = agent.init_state(N), 0
    while step < args.steps:
        start_state = {k: v.clone() for k, v in state.items()}
        seg = {"obs": [], "action": [], "reward": [], "done": []}
        with torch.no_grad():
            for _ in range(T):
                dist, _, state = agent.step(obs, state, done)
                action = dist.sample()
                raw, reward, term, trunc, _ = envs.step(action.numpy())
                tracker.update(reward, term | trunc, step); step += N
                seg["obs"].append(obs); seg["action"].append(action); seg["done"].append(done)
                seg["reward"].append(torch.as_tensor(reward, dtype=torch.float32))
                obs, done = observe(raw), torch.as_tensor(term | trunc, dtype=torch.float32)
            _, next_value, _ = agent.step(obs, {k: v.clone() for k, v in state.items()}, done)
        seg = {k: torch.stack(v) for k, v in seg.items()}
        returns, ret = torch.zeros(T, N), next_value
        for t in reversed(range(T)):
            ended = done if t == T - 1 else seg["done"][t + 1]
            ret = seg["reward"][t] + args.gamma * ret * (1 - ended)
            returns[t] = ret
        agent.core.clear_rates()
        dist, values, _ = agent.seq(seg["obs"], start_state, seg["done"])
        advantage = (returns - values).detach()
        loss = (-(agent.log_prob(dist, seg["action"]) * advantage).mean()
                + 0.5 * ((returns - values) ** 2).mean() + agent.core.homeostasis_loss())
        optimizer.zero_grad(); loss.backward(); optimizer.step()


# ============================================================================ three-factor rules (no backprop)
def agent_step_local(obs, state, done, learn_hidden, exact=False):
    """Encoder + DendraCore.step, cut into blocks: the encoder, the input synapses, then each cortical region.
    Every block receives a DETACHED input, so a learning signal at a block's output can only change that
    block's own synapses. Returns the final activity h (detached), the output of each block (with its local
    graph), the firing rates and the new state. exact=True: no cuts (true backprop inside one step)."""
    cut = (lambda v: v) if exact else (lambda v: v.detach())
    core = agent.core
    keep = 1 - done
    S = state["S"] * keep[None, :, None, None, None]
    m, t = state["m"] * keep[:, None], state["t"] * keep
    with torch.set_grad_enabled(learn_hidden):
        features = agent.encoder(obs)
        x_in = core.inp(cut(features))
    with torch.no_grad():                                   # neuromodulators: fixed (gain = ACh = 1 at init)
        m = core.nm.decay * m + x_in
        gain, ach = core.neuromodulator_signals(m, t)
    outputs, h, new_S, rates = [features, x_in], cut(x_in)[:, None], [], []
    for l, region in enumerate(core.regions):
        with torch.set_grad_enabled(learn_hidden):
            h_out, S_l = region.step(h, gain[:, None, l], ach[:, None, l], None, S[l])
        outputs.append((h_out - h)[:, 0])                   # what this region added to the residual stream
        rates.append(region.l5.last_rate)
        h = cut(h_out)
        new_S.append(S_l.detach())                          # the Hebbian memory is state, not a trained weight
    new_state = {"S": torch.stack(new_S), "m": m, "t": t + 1}
    return h[:, 0], outputs, rates, new_state


class LocalForward(nn.Module):
    """One step of the whole agent (blocks + readouts), so it can run on a snapshot of the weights."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, obs, state, done, learn_hidden, exact):
        h, outputs, rates, new_state = agent_step_local(obs, state, done, learn_hidden, exact)
        return h, outputs, rates, new_state, self.agent.policy_head(h), self.agent.value_head(h)[:, 0]


def train_three_factor(tracker, learn_hidden, exact=False):
    core = agent.core
    for p in core.nm.parameters():
        p.requires_grad_(False)
    if not learn_hidden:                                    # only the two readouts learn
        for p in list(core.parameters()) + list(agent.encoder.parameters()):
            p.requires_grad_(False)
    value_params = list(agent.value_head.parameters())
    policy_params = list(agent.policy_head.parameters())
    hidden_params = [p for p in list(agent.encoder.parameters()) + list(core.parameters()) if p.requires_grad]
    # The critic learns faster than the actor (otherwise every early step looks "better than expected" and the
    # policy collapses onto whatever it happened to do). Adam's step per synapse uses only that synapse's history.
    groups = [{"params": policy_params, "lr": lr}, {"params": value_params, "lr": lr * args.critic_mult}]
    if hidden_params:
        groups.append({"params": hidden_params, "lr": lr * args.hidden_mult})
    optimizer = torch.optim.Adam(groups)
    d, A = core.d, 2
    gen = torch.Generator().manual_seed(1000 + args.seed)
    widths = [64] + [d] * (1 + len(core.regions))          # outputs of: encoder, input synapses, each region
    feedback_B = [torch.randn(A, w, generator=gen) / w ** 0.5 for w in widths]   # fixed, random
    feedback_b = [torch.randn(w, generator=gen) / w ** 0.5 for w in widths]

    stepper = LocalForward(agent)
    obs, done = observe(envs.reset(seed=args.seed)[0]), torch.zeros(N)
    state, step, previous = core.init_state(N), 0, None
    while step < args.steps:
        # Run the step on a copy of the weights: the update for the previous step (below) changes the weights
        # in place while this step's eligibility is still pending. Same as applying each update one step later.
        snapshot = {name: p.clone() for name, p in stepper.named_parameters()}
        h, outputs, rates, state, logits, value = torch.func.functional_call(
            stepper, snapshot, (obs, state, done, learn_hidden, exact))   # h is detached unless exact
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        if previous is not None:                            # dopamine for the previous step, now that V(s_t+1) is known
            p_logp, p_value, p_outputs, p_error, p_rates, p_reward = previous
            delta = p_reward + args.gamma * value.detach() * (1 - done) - p_value.detach()   # [N]
            loss = -(delta * p_logp).mean() - 0.5 * (delta * p_value).mean()   # readouts: lr * delta * (pre x post error)
            if learn_hidden and not exact:                  # hidden synapses: delta x (block output . random feedback)
                for y, B, b in zip(p_outputs, feedback_B, feedback_b):
                    post_error = p_error @ B + 0.5 * b      # [N, d]
                    loss = loss - (delta[:, None] * y * post_error).sum(-1).mean()
            if learn_hidden:
                loss = loss + ((torch.stack(p_rates) - 0.15) ** 2).sum()   # homeostasis, local to each region
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        raw, reward, term, trunc, _ = envs.step(action.numpy())
        tracker.update(reward, term | trunc, step); step += N
        error = F.one_hot(action, A).float() - dist.probs.detach()            # post-synaptic error of the policy
        previous = (dist.log_prob(action), value, outputs, error, rates, torch.as_tensor(reward, dtype=torch.float32))
        obs, done = observe(raw), torch.as_tensor(term | trunc, dtype=torch.float32)


tracker, start = Tracker(), time.time()
if args.method == "backprop":
    train_backprop(tracker)
else:
    train_three_factor(tracker, learn_hidden=(args.method in ("eprop", "online_backprop")),
                       exact=(args.method == "online_backprop"))
elapsed = time.time() - start
last = [r for _, r in tracker.finished[-50:]]
result = {"method": args.method, "task": args.task, "seed": args.seed, "lr": lr, "critic_mult": args.critic_mult, "hidden_mult": args.hidden_mult, "steps": args.steps,
          "final_return": float(np.mean(last)) if last else None, "episodes": tracker.finished, "time_s": elapsed}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(result, f)
print(f"{args.method:8} {args.task:15} seed {args.seed} lr {lr:g}: mean return of the last 50 episodes "
      f"{result['final_return']:.1f} | {len(tracker.finished)} episodes | {elapsed:.0f} s", flush=True)
