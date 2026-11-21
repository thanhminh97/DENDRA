"""
Atari results for the visual DENDRA-Brain vs PPO-CNN: table (with ablations) and learning curves.

Score = mean return of the last 20 episodes of training (100k environment steps). Published Atari-100k scores
of deep-RL methods are listed for reference; they use other evaluation protocols (separate evaluation episodes,
1 environment), so they are only a rough yardstick.

Usage: python experiments/rl/plot_atari.py [results folder] [output image]
"""
import glob
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/brain_atari"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "figures/brain_atari.png"
warnings.filterwarnings("ignore", "Mean of empty slice")

GAMES = ["Breakout", "Boxing", "Freeway"]
AGENTS = {
    "ppo": ("PPO-CNN (4 stacked frames)", "#9a9a9a", "--"),
    "full": ("DENDRA-Brain, all modules", "#2a6fdb", "-"),
    "noplan": ("  - planning (replay)", "#7fb3f5", "-"),
    "nomem": ("  - hippocampal memory", "#a0c4f8", ":"),
    "noconj": ("  - dendritic conjunctions", "#c6dcfb", "-."),
}
# random / human: Atari-5 paper (Aitchison et al., 2022); SPR and Rainbow (enhanced): table in arXiv:2606.21297
REFERENCE = {"Breakout": (1.7, 30.5, 13, 14), "Boxing": (0.1, 12.1, 38, 28), "Freeway": (0.0, 29.6, 21, 6)}


def final(result, window=20):
    returns = [r for _, r in result["episodes"]]
    return float(np.mean(returns[-window:])) if returns else np.nan


def curve(result, grid, window=20):
    episodes = np.array(result["episodes"], dtype=float).reshape(-1, 2)
    return np.array([np.mean(episodes[episodes[:, 0] <= s][-window:, 1]) if (episodes[:, 0] <= s).any() else np.nan
                     for s in grid])


def main():
    runs = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        kind, game = re.match(r"(\w+?)_(\w+)_s\d+\.json", os.path.basename(path)).groups()
        runs[game][kind].append(json.load(open(path)))
    print(f"{'Game':10} {'Agent':36} {'Score (last 20 episodes)':>26} {'Steps/s':>8}  per seed")
    fig, axes = plt.subplots(1, len(GAMES), figsize=(5.2 * len(GAMES), 3.8), squeeze=False)
    for ax, game in zip(axes[0], GAMES):
        grid = np.linspace(5000, 100_000, 40)
        for kind, (label, color, style) in AGENTS.items():
            results = runs[game].get(kind, [])
            if not results:
                continue
            f = np.array([final(r) for r in results])
            speed = np.mean([(r.get("steps") or r.get("total")) / r["time_s"] for r in results])
            print(f"{game:10} {label:36} {f.mean():14.2f} ± {f.std():5.2f} {speed:8.0f}  ({', '.join(f'{x:.1f}' for x in f)})")
            if kind in ("ppo", "full"):
                curves = np.array([curve(r, grid) for r in results])
                ax.plot(grid, np.nanmean(curves, 0), style, color=color, lw=1.8, label=label.strip())
                if len(results) > 1:
                    ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=color, alpha=0.12)
        rnd, human, spr, rainbow = REFERENCE[game]
        print(f"{game:10} {'(reference) random / human':36} {rnd:14.1f} / {human:.1f}")
        print(f"{game:10} {'(published, 100k) SPR / Rainbow-enh.':36} {spr:14.1f} / {rainbow:.1f}")
        ax.axhline(rnd, color="#cccccc", ls="--", lw=0.8)
        ax.text(100_000, rnd, " random", fontsize=7, color="#999999", va="bottom", ha="right")
        ax.set_title(game, loc="left", fontsize=11)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return (avg. of 20)")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Figure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
