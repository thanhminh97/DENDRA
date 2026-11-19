"""Draw the language-modeling figures of the README from experiments/results/language/*.json.

Run from the repository root:  python experiments/make_figures.py
(The RL figures are drawn by experiments/rl/plot_bandit.py and plot_ppo.py.)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "experiments/results/language"
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
GREY, BLUE = "#9a9a9a", "#2a6fdb"

# 1) training loss curves (per step)
runs = [("gpt_long", "GPT, 3,000 steps (~ same CPU time as DENDRA 1,200 steps)", GREY, "--"),
        ("gpt", "GPT, 1,200 steps", GREY, "-"),
        ("dendra_fb1", "DENDRA", BLUE, "-"),
        ("dendra_fb2", "DENDRA + 2 feedback passes", BLUE, "--"),
        ("dendra_spikein", "DENDRA + spiking input", BLUE, ":"),
        ("dendra_nonm", "DENDRA - neuromodulators", "#8fb4ee", "-.")]
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, label, c, ls in runs:
    f = f"{R}/{name}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    ax.plot([p["step"] for p in r["log"]], [p["loss"] for p in r["log"]], ls, color=c, label=label, lw=1.6)
ax.set_xlabel("training step"); ax.set_ylabel("training loss (nats/char)")
ax.set_title("At equal steps DENDRA learns faster; GPT needs ~1,500 steps to match DENDRA at step 1,200", fontsize=10.5, loc="left")
ax.set_ylim(1.2, 2.8); ax.set_xlim(0, 3000); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig("figures/training_curves.png", dpi=150); plt.close()

# 2) inference speed and memory
b = json.load(open(f"{R}/bench_cpu.json"))["rows"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k, label, c, mk in [("gpt", "GPT (KV-cache)", GREY, "s"), ("dendra", "DENDRA (constant-size state)", BLUE, "o")]:
    rs = [r for r in b if r["model"] == k]
    x = [r["ctx"] for r in rs]
    ax[0].plot(x, [r["decode_ms"] for r in rs], marker=mk, color=c, label=label)
    ax[1].plot(x, [r["prefill_ms"] / 1000 for r in rs], marker=mk, color=c, label=label)
    ax[2].plot(x, [r["state_mb"] for r in rs], marker=mk, color=c, label=label)
for a, yl, t in zip(ax, ["ms / token", "seconds", "MB / conversation"],
                    ["Generate one token", "Read prompt (prefill)", "State memory"]):
    a.set_xscale("log", base=2); a.set_xlabel("context length (tokens)"); a.set_ylabel(yl)
    a.set_title(t, fontsize=11, loc="left"); a.grid(alpha=0.25)
ax[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig("figures/inference_speed.png", dpi=150); plt.close()
# 3) speed on a T4 GPU (Kaggle)
f = f"{R}/bench_t4_kaggle.json"
if os.path.exists(f):
    b = json.load(open(f))["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, c, mk in [("gpt", GREY, "s"), ("dendra", BLUE, "o")]:
        for bs, ls in [(1, "-"), (16, "--")]:
            rs = [r for r in b if r["model"] == k and r["batch"] == bs]
            lab = f"{'GPT' if k == 'gpt' else 'DENDRA'}, batch {bs}"
            ax[0].plot([r["ctx"] for r in rs], [r["decode_ms"] for r in rs], ls, marker=mk, color=c, label=lab)
        rs = [r for r in b if r["model"] == k and r["batch"] == 1]
        ax[1].plot([r["ctx"] for r in rs], [r["mem_mb_per_seq"] for r in rs], marker=mk, color=c,
                   label="GPT (KV-cache)" if k == "gpt" else "DENDRA (constant-size state)")
    for a, yl, t in zip(ax, ["ms / decode step", "MB / conversation"], ["Decoding on a T4: GPT is faster at every measured length", "Memory per conversation"]):
        a.set_xscale("log", base=2); a.set_xlabel("context length (tokens)"); a.set_ylabel(yl)
        a.set_title(t, fontsize=10.5, loc="left"); a.grid(alpha=0.25); a.legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig("figures/inference_speed_t4.png", dpi=150); plt.close()
print("ok")
