"""Plot PPO learning curves and print a summary table from runs/ppo_*.json."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "runs"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/rl_ppo.png"
STYLE = {"mlp": ("Memoryless (MLP/CNN)", "#9a9a9a", "--"), "lstm": ("LSTM", "#555555", ":"),
         "lstm_big": ("Large LSTM (same #params)", "#222222", "-."), "dendra": ("DENDRA", "#2a6fdb", "-")}
TITLE = {"minatar:breakout": "MinAtar Breakout (discrete)", "pendulum-pomdp": "Pendulum, velocity hidden (continuous)",
         "atari:Breakout": "Atari Breakout (discrete)", "atari:Pong": "Atari Pong (discrete)",
         "mujoco:HalfCheetah-v5-pomdp": "HalfCheetah, velocities hidden (continuous)", "mujoco:HalfCheetah-v5": "HalfCheetah (continuous)"}
runs = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{src}/ppo_*.json"):
    r = json.load(open(f))
    runs[r["env"]][r["core"]].append(r)


def smooth_curve(r, grid, win=100):
    """Moving average of finished-episode returns, sampled on a grid of environment steps."""
    ep = np.array(r["episodes"]) if r["episodes"] else np.zeros((0, 2))
    out = []
    for g in grid:
        rr = ep[ep[:, 0] <= g][-win:, 1] if len(ep) else []
        out.append(np.mean(rr) if len(rr) else np.nan)
    return np.array(out)


envs = sorted(runs)
fig, axes = plt.subplots(1, len(envs), figsize=(6.5 * len(envs), 4), squeeze=False)
plt.rcParams.update({"font.size": 10})
print(f"{'Environment':38} {'Core':24} {'Params':>8} {'Return, last 100 episodes':>26} {'Steps/s':>10}")
for ax, env in zip(axes[0], envs):
    total = max(r["total"] for rs in runs[env].values() for r in rs)
    grid = np.linspace(total / 50, total, 50)
    for core in ("mlp", "lstm", "lstm_big", "dendra"):
        if core not in runs[env]: continue
        lab, c, ls = STYLE[core]
        curves = np.array([smooth_curve(r, grid) for r in runs[env][core]])
        m = np.nanmean(curves, 0)
        ax.plot(grid, m, ls, color=c, lw=1.8, label=f"{lab} ({len(curves)} seed{"s" if len(curves) > 1 else ""})")
        if len(curves) > 1:
            ax.fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=c, alpha=0.12)
        fin = [r["final_return_mean"] if r["final_return_mean"] is not None else np.nan for r in runs[env][core]]
        sps = [r["total"] / r["time_s"] for r in runs[env][core]]
        print(f"{env:38} {lab:24} {runs[env][core][0]['params']/1e3:7.0f}K {np.nanmean(fin) if not np.all(np.isnan(fin)) else np.nan:12.2f} ± {np.nanstd(fin) if not np.all(np.isnan(fin)) else np.nan:6.2f} {np.mean(sps):10.0f}")
    ax.set_title(TITLE.get(env, env), loc="left", fontsize=11)
    ax.set_xlabel("environment steps"); ax.set_ylabel("episode return (moving avg. of 100)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
plt.tight_layout()
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
plt.savefig(out, dpi=150)
