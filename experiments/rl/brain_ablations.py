"""
Ablation table for DENDRA-Brain: each row removes one mechanism (100k steps, seeds 10-12).

Usage: python experiments/rl/brain_ablations.py [ablation folder] [main results folder]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ABL_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/brain/ablations_100k"
MAIN_DIR = sys.argv[2] if len(sys.argv) > 2 else "experiments/results/brain/main_100k"
TASKS = {"CartPole-v1-pomdp": "CartPole, vel. hidden", "Acrobot-v1": "Acrobot",
         "MountainCar-v0": "MountainCar", "Pendulum-v1-pomdp": "Pendulum, vel. hidden"}
VARIANTS = {
    "": "Full DENDRA-Brain",
    "timescalesnone": "- temporal context (time cells)",
    "lam0": "- eligibility traces (lambda = 0)",
    "adaptiveFalse": "- adaptive dopamine coding",
    "explore0": "- action-novelty bonus",
    "field1000": "- receptive-field gating (k always fire)",
    "areas1subspace100primary_max0": "- columns (one area sees everything)",
}


def scores(path):
    episodes = json.load(open(path))["episodes"]
    returns = [r for _, r in episodes]
    return np.mean(returns[-50:]), np.mean(returns)          # final, mean over the whole run


results = defaultdict(list)
for path in glob.glob(os.path.join(ABL_DIR, "*.json")):
    m = re.match(r"brain(.*)_(.*)_s\d+\.json", os.path.basename(path))
    results[(m.group(1), m.group(2))].append(scores(path))
for path in glob.glob(os.path.join(MAIN_DIR, "brain_*.json")):
    m = re.match(r"brain_(.*)_s\d+\.json", os.path.basename(path))
    results[("", m.group(1))].append(scores(path))

print("Final return (last 50 episodes) / mean return over the whole run, mean over seeds")
print(f"{'Variant':44}" + "".join(f"{name:>26}" for name in TASKS.values()))
for key, label in VARIANTS.items():
    row = []
    for task in TASKS:
        s = np.array(results.get((key, task), []))
        row.append(f"{s[:, 0].mean():9.1f} / {s[:, 1].mean():8.1f}" if len(s) else "-")
    print(f"{label:44}" + "".join(f"{x:>26}" for x in row))
