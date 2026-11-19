"""
Summary table and figure for experiments/rl/three_factor.py.

For each method the configuration (learning rates) is chosen on CartPole (best mean over seeds);
the same configuration is then used, unchanged, on CartPole with hidden velocities.

Usage:
  python experiments/rl/plot_three_factor.py [results folder] [output image]
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

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/rl_three_factor"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "figures/rl_three_factor.png"
warnings.filterwarnings("ignore", "Mean of empty slice")

METHODS = {
    "backprop": ("Backprop (A2C + BPTT)", "#9a9a9a", "--"),
    "online_backprop": ("Backprop, online (same schedule as e-prop)", "#444444", ":"),
    "readout": ("Three-factor, output synapses only", "#7fb3f5", "-"),
    "eprop": ("Three-factor, all synapses (e-prop-like)", "#2a6fdb", "-"),
}
TASKS = {"cartpole": "CartPole", "cartpole-pomdp": "CartPole, velocities hidden"}


def load(folder):
    """runs[task][method][(lr, hidden_mult)] = list of results (one per seed)."""
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path) as f:
            r = json.load(f)
        runs[r["task"]][r["method"]][(r["lr"], r.get("hidden_mult", 1.0))].append(r)
    return runs


def final_returns(results):
    return np.array([r["final_return"] for r in results], dtype=float)


def curve(result, grid, window=50):
    episodes = np.array(result["episodes"])
    out = []
    for step in grid:
        recent = episodes[episodes[:, 0] <= step][-window:, 1]
        out.append(recent.mean() if len(recent) else np.nan)
    return np.array(out)


def main():
    runs = load(RESULTS_DIR)
    best = {}
    print("Configurations on CartPole (mean return of the last 50 episodes, mean ± std over seeds):")
    for method in METHODS:
        configs = runs["cartpole"].get(method, {})
        for (lr, hm), results in sorted(configs.items()):
            f = final_returns(results)
            extra = f", hidden x{hm:g}" if method in ("eprop", "online_backprop") else ""
            print(f"  {method:9} lr {lr:g}{extra:16} {f.mean():7.1f} ± {f.std():5.1f}  ({len(f)} seeds)")
        if configs:
            best[method] = max(configs, key=lambda c: final_returns(configs[c]).mean())

    tasks = [t for t in TASKS if t in runs]
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4), squeeze=False)
    print("\nChosen configuration per method (selected on CartPole, reused unchanged on the hidden-velocity task):")
    for ax, task in zip(axes[0], tasks):
        total = max(r["steps"] for m in runs[task].values() for rs in m.values() for r in rs)
        grid = np.linspace(total / 40, total, 40)
        for method, (label, color, style) in METHODS.items():
            if method not in best or best[method] not in runs[task].get(method, {}):
                continue
            results = runs[task][method][best[method]]
            curves = np.array([curve(r, grid) for r in results])
            ax.plot(grid, np.nanmean(curves, 0), style, color=color, lw=1.8, label=label)
            ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=color, alpha=0.12)
            f = final_returns(results)
            lr, hm = best[method]
            print(f"  {TASKS[task]:28} {label:42} lr {lr:g}{f', hidden x{hm:g}' if method in ('eprop', 'online_backprop') else '':14} "
                  f"{f.mean():7.1f} ± {f.std():5.1f}  per seed: {', '.join(f'{x:.0f}' for x in f)}")
        ax.axhline(22, color="#cccccc", lw=0.8, ls="--")
        ax.text(total, 22, " random play", color="#999999", fontsize=7, va="bottom", ha="right")
        ax.set_title(TASKS[task], loc="left", fontsize=11)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return (moving avg. of 50)")
        ax.set_ylim(0, 520 if task == "cartpole" else 80)    # the hidden-velocity panel is zoomed in
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nFigure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
