"""
Train the visual DENDRA-Brain (dendra/visual_brain.py) online on an Atari game and save the episode returns.

  python experiments/rl/run_visual_brain.py --game Breakout --steps 100000 --seed 0 --out runs/vb.json
  Agent settings: --set lr=0.1 episodic=1.0 conjunctions=False ...
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
from dendra.visual_brain import VisualBrainAgent
from envs import VecEnv

parser = argparse.ArgumentParser()
parser.add_argument("--game", default="Breakout")
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--set", nargs="*", default=[])
parser.add_argument("--out", default="runs/visual_brain.json")
args = parser.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(1)


def parse(value):
    if value in ("True", "False"):
        return value == "True"
    return int(value) if value.lstrip("-").isdigit() else float(value)


settings = {k: parse(v) for k, v in (item.split("=") for item in args.set)}
env = VecEnv("atari:" + args.game, args.num_envs, args.seed)
agent = VisualBrainAgent(env.n_actions, n_envs=args.num_envs, seed=args.seed, **settings)

action = agent.reset(env.reset())
running, episodes, start, step = np.zeros(args.num_envs), [], time.time(), 0
while step < args.steps:
    obs, reward, done = env.step(action)
    running += reward
    for i in np.nonzero(done)[0]:
        episodes.append((step, float(running[i]))); running[i] = 0
    step += args.num_envs
    action = agent.step(obs, reward, done)
    if step % (args.steps // 10) < args.num_envs:
        recent = [r for _, r in episodes[-20:]]
        print(f"step {step:7d} | mean return (last 20 episodes) {np.mean(recent) if recent else float('nan'):7.2f} | "
              f"{len(episodes)} episodes | {step / (time.time() - start):4.0f} steps/s", flush=True)

last = [r for _, r in episodes[-20:]]
result = {"agent": "visual_brain", "env": "atari:" + args.game, "seed": args.seed, "settings": settings, "steps": step,
          "final_return": float(np.mean(last)) if last else None, "episodes": episodes, "time_s": time.time() - start}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(result, open(args.out, "w"))
print(f"DONE {args.game} seed {args.seed}: mean return (last 20 episodes) "
      f"{result['final_return'] if last else float('nan'):.2f}, {len(episodes)} episodes, {result['time_s']:.0f} s", flush=True)
