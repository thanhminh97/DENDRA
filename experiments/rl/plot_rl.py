"""Plot meta-RL bandit results from runs/rl_*_s*.json (or experiments/results/rl/)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "runs"
NAMES = [("lstm", "LSTM", "#9a9a9a", "--"), ("gpt", "GPT (Transformer)", "#555555", ":"),
         ("dendra", "DENDRA", "#2a6fdb", "-"), ("dendra_da", "DENDRA + dopamine", "#0b3d91", "-")]
runs = {}
for k, *_ in NAMES:
    fs = sorted(glob.glob(f"{src}/rl_{k}_s*.json"))
    if fs:
        runs[k] = [json.load(open(f)) for f in fs]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# 1) training curves
for k, lab, c, ls in NAMES:
    if k not in runs: continue
    u = [p["update"] for p in runs[k][0]["log"]]
    r = np.mean([[p["regret"] for p in rr["log"]] for rr in runs[k]], 0)
    ax[0].plot(u, r, ls, color=c, label=lab, lw=1.6)
ax[0].set_xlabel("updates"); ax[0].set_ylabel("end-of-episode regret (during training)")
ax[0].set_title("Learning speed", loc="left"); ax[0].grid(alpha=0.25); ax[0].legend(fontsize=8, frameon=False)

# 2) cumulative regret within one episode (T = 100)
any_run = next(iter(runs.values()))[0]
for kb, lab, c in [("thompson", "Thompson Sampling", "#2e9e5b"), ("ucb1", "UCB1", "#d08b1f"), ("random", "Random", "#cccccc")]:
    ax[1].plot(any_run["eval"]["baseline_curves"][kb], color=c, lw=1.2, label=lab)
for k, lab, c, ls in NAMES:
    if k not in runs: continue
    ax[1].plot(np.mean([rr["eval"]["regret_curve"] for rr in runs[k]], 0), ls, color=c, lw=1.8, label=lab)
ax[1].set_xlabel("pull within the episode"); ax[1].set_ylabel("cumulative regret")
ax[1].set_title("In-context learning (T = 100)", loc="left"); ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_ylim(0, 16)

# 3) episodes 3x longer than in training
for k, lab, c, ls in NAMES:
    if k not in runs: continue
    ax[2].plot(np.mean([rr["eval_long"]["regret_curve"] for rr in runs[k]], 0), ls, color=c, lw=1.8, label=lab)
ax[2].plot(any_run["eval_long"]["baseline_curves"]["thompson"], color="#2e9e5b", lw=1.2, label="Thompson Sampling")
ax[2].axvline(100, color="#bbbbbb", lw=1, ls="--"); ax[2].text(104, 1, "end of training length", fontsize=8, color="#888888")
ax[2].set_xlabel("pull within the episode"); ax[2].set_ylabel("cumulative regret")
ax[2].set_title("3x longer episodes (T = 300, never trained on)", loc="left"); ax[2].grid(alpha=0.25); ax[2].legend(fontsize=8, frameon=False)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/rl_bandit.png", dpi=150)

print(f"{'Agent':22} {'params':>8} {'regret T=100':>14} {'regret T=300':>14} {'s/update':>14} {'ms/decision':>14}")
for k, lab, *_ in NAMES:
    if k not in runs: continue
    rs = runs[k]
    r100 = [r["eval"]["final_regret"] for r in rs]; r300 = [r["eval_long"]["final_regret"] for r in rs]
    print(f"{lab:22} {rs[0]['params']/1e3:7.0f}K {np.mean(r100):8.2f} ±{np.std(r100):4.2f} {np.mean(r300):8.2f} ±{np.std(r300):4.2f} "
          f"{np.mean([r['train_time_s']/(r['updates']+1) for r in rs]):14.2f} {np.mean([r['decision_ms'] for r in rs]):14.2f}")
b = any_run["eval"]["baselines"]; bl = any_run["eval_long"]["baselines"]
for kb in ("thompson", "ucb1", "random"):
    print(f"{kb:22} {'':>8} {b[kb]:14.2f} {bl[kb]:14.2f}")
