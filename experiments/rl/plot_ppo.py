"""
Plot PPO learning curves and print a summary table from the runs/ppo_*.json files
written by ppo.py.

Usage:
  python experiments/rl/plot_ppo.py [results folder, default runs] [output image, default figures/rl_ppo.png]
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

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "figures/rl_ppo.png"
warnings.filterwarnings("ignore", "Mean of empty slice")   # early points where no episode has finished yet

# core name -> (legend label, color, line style); the order here is also the drawing order
STYLE = {
    "mlp": ("Memoryless (MLP/CNN)", "#9a9a9a", "--"),
    "lstm": ("LSTM", "#555555", ":"),
    "lstm_big": ("Large LSTM (same #params)", "#222222", "-."),
    "dendra": ("DENDRA", "#2a6fdb", "-"),
}
TITLES = {
    "minatar:breakout": "MinAtar Breakout (discrete)",
    "pendulum-pomdp": "Pendulum, velocity hidden (continuous)",
    "atari:Breakout": "Atari Breakout (discrete)",
    "atari:Pong": "Atari Pong (discrete)",
    "mujoco:HalfCheetah-v5-pomdp": "HalfCheetah, velocities hidden (continuous)",
    "mujoco:HalfCheetah-v5": "HalfCheetah (continuous)",
}


def load_runs(folder):
    """runs[env][core] = list of result dicts (one per seed)."""
    runs = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(folder, "ppo_*.json")):
        with open(path) as f:
            result = json.load(f)
        runs[result["env"]][result["core"]].append(result)
    return runs


def smoothed_curve(result, grid, window=100):
    """Mean return of the last `window` finished episodes, read at every step count in `grid`."""
    episodes = np.array(result["episodes"]) if result["episodes"] else np.zeros((0, 2))
    curve = []
    for step in grid:
        recent = episodes[episodes[:, 0] <= step][-window:, 1]
        curve.append(recent.mean() if len(recent) else np.nan)
    return np.array(curve)


def mean_and_std(values):
    values = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    return np.nanmean(values), np.nanstd(values)


def main():
    runs = load_runs(RESULTS_DIR)
    if not runs:
        sys.exit(f"No ppo_*.json files found in {RESULTS_DIR}/")
    envs = sorted(runs)
    fig, axes = plt.subplots(1, len(envs), figsize=(6.5 * len(envs), 4), squeeze=False)
    print(f"{'Environment':38} {'Core':26} {'Params':>8} {'Return, last 100 episodes':>26} {'Steps/s':>9}")

    for ax, env in zip(axes[0], envs):
        total_steps = max(r["total"] for results in runs[env].values() for r in results)
        grid = np.linspace(total_steps / 50, total_steps, 50)
        for core, (label, color, line_style) in STYLE.items():
            results = runs[env].get(core)
            if not results:
                continue
            curves = np.array([smoothed_curve(r, grid) for r in results])
            n_seeds = len(curves)
            seeds_text = f"{n_seeds} seed" + ("s" if n_seeds > 1 else "")
            ax.plot(grid, np.nanmean(curves, 0), line_style, color=color, lw=1.8, label=f"{label} ({seeds_text})")
            if n_seeds > 1:                                   # shaded band: min..max over seeds
                ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=color, alpha=0.12)

            mean, std = mean_and_std([r["final_return_mean"] for r in results])
            steps_per_s = np.mean([r["total"] / r["time_s"] for r in results])
            params = results[0]["params"] / 1e3
            print(f"{env:38} {label:26} {params:7.0f}K {mean:16.2f} ± {std:7.2f} {steps_per_s:9.0f}")

        ax.set_title(TITLES.get(env, env), loc="left", fontsize=11)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return (moving avg. of 100)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Figure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
