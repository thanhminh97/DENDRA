"""
Train DENDRA-Brain (dendra/brain.py) online on one environment and save the episode returns.

  python experiments/rl/run_brain.py --env CartPole-v1 --steps 100000 --seed 0 --out runs/brain.json
  Extra agent settings: --set lr=0.5 temperature=0.1 replay=1 ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dendra.brain import BrainAgent
from envs import VecEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v1")
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--set", nargs="*", default=[], help="agent settings, e.g. lr=0.5 replay=1")
parser.add_argument("--out", default="runs/brain.json")
args = parser.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(1)

settings = {}
for item in args.set:
    key, value = item.split("=")
    settings[key] = () if value == "none" else tuple(float(v) for v in value.split(",") if v) if "," in value else (int(value) if value.isdigit() else (value == "True") if value in ("True", "False") else float(value))

env = VecEnv(args.env, args.num_envs, args.seed)
obs_dim = int(np.prod(env.obs_shape))
if env.continuous:
    agent = BrainAgent(obs_dim, action_low=env.low, action_high=env.high, n_envs=args.num_envs, seed=args.seed, **settings)
else:
    agent = BrainAgent(obs_dim, n_actions=env.n_actions, n_envs=args.num_envs, seed=args.seed, **settings)

obs = env.reset()
action = agent.reset(obs)
running, episodes, start, step = np.zeros(args.num_envs), [], time.time(), 0
while step < args.steps:
    obs, reward, done = env.step(action)
    running += reward
    for i in np.nonzero(done)[0]:
        episodes.append((step, float(running[i]))); running[i] = 0
    step += args.num_envs
    action = agent.step(obs, reward, done)
    if step % (args.steps // 10) < args.num_envs:
        recent = [r for _, r in episodes[-50:]]
        print(f"step {step:7d} | mean return (last 50) {np.mean(recent) if recent else float('nan'):8.1f} | "
              f"units {agent.cortex.recruited} | {step / (time.time() - start):5.0f} steps/s", flush=True)

last = [r for _, r in episodes[-50:]]
result = {"agent": "brain", "env": args.env, "seed": args.seed, "settings": settings, "steps": step,
          "final_return": float(np.mean(last)) if last else None, "episodes": episodes, "time_s": time.time() - start}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(result, open(args.out, "w"))
print(f"DONE {args.env} seed {args.seed}: mean return (last 50 episodes) {result['final_return']:.1f}, "
      f"{len(episodes)} episodes, {result['time_s']:.0f} s", flush=True)
