"""
Compare DENDRA-Brain with PPO baselines: summary table and learning curves.

Reads every *.json in the folder (written by run_brain.py or ppo.py). For each task and agent:
  final  mean return of the last 50 episodes (mean ± std over seeds)
  AUC    mean of the learning curve over the whole run (sample efficiency: higher = learned sooner)

Usage: python experiments/rl/plot_brain.py [results folder] [output image]
"""
import glob
import json
import os
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/brain"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "figures/brain_vs_ppo.png"
warnings.filterwarnings("ignore", "Mean of empty slice")

AGENTS = {
    "ppo_mlp": ("PPO, MLP (frame stack if velocities hidden)", "#9a9a9a", "--"),
    "ppo_lstm": ("PPO, LSTM", "#444444", ":"),
    "brain": ("DENDRA-Brain (online, local rules)", "#2a6fdb", "-"),
}
TASKS = {
    "CartPole-v1": "CartPole",
    "CartPole-v1-pomdp": "CartPole, velocities hidden",
    "Acrobot-v1": "Acrobot",
    "MountainCar-v0": "MountainCar (sparse reward)",
    "Pendulum-v1": "Pendulum (continuous)",
    "Pendulum-v1-pomdp": "Pendulum, velocity hidden (continuous)",
}


def agent_of(result):
    return "brain" if result.get("agent") == "brain" else "ppo_" + result["core"]


def task_of(result):
    return result["env"].replace("classic:", "")


def load(folder):
    runs = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path) as f:
            r = json.load(f)
        runs[task_of(r)][agent_of(r)].append(r)
    return runs


def curve(result, grid, window=50):
    episodes = np.array(result["episodes"], dtype=float).reshape(-1, 2)
    out = []
    for step in grid:
        recent = episodes[episodes[:, 0] <= step][-window:, 1]
        out.append(recent.mean() if len(recent) else np.nan)
    return np.array(out)


def final(result, window=50):
    episodes = result["episodes"]
    return float(np.mean([r for _, r in episodes[-window:]])) if episodes else np.nan


def main():
    runs = load(RESULTS_DIR)
    tasks = [t for t in TASKS if t in runs]
    cols = 3
    rows = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)
    print(f"{'Task':40} {'Agent':44} {'Final (last 50 ep.)':>22} {'AUC':>9} {'Steps/s':>8} seeds")
    for ax, task in zip(axes.flat, tasks):
        total = max((r.get("steps") or r.get("total")) for rs in runs[task].values() for r in rs)
        grid = np.linspace(total / 50, total, 50)
        for agent, (label, color, style) in AGENTS.items():
            results = runs[task].get(agent, [])
            if not results:
                continue
            curves = np.array([curve(r, grid) for r in results])
            mean = np.nanmean(curves, 0)
            ax.plot(grid, mean, style, color=color, lw=1.8, label=label)
            if len(results) > 1 and np.isfinite(curves).any():
                ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=color, alpha=0.12)
            finals = np.array([final(r) for r in results])
            auc = np.nanmean(curves)
            speed = np.mean([(r.get("steps") or r.get("total")) / r["time_s"] for r in results])
            print(f"{TASKS[task]:40} {label:44} {finals.mean():12.1f} ± {finals.std():6.1f} {auc:9.1f} {speed:8.0f} "
                  f"{len(results)}  ({', '.join(f'{x:.0f}' for x in finals)})")
        ax.set_title(TASKS[task], loc="left", fontsize=10)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("return (moving avg. of 50 episodes)", fontsize=8)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Figure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
