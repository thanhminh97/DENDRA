"""Inference speed (no training): DENDRA vs a same-size GPT, on CPU or GPU.

Prefill  : time to read the whole prompt (DENDRA reads in chunks, GPT uses causal attention).
Decode   : mean time per new token when the context already holds n tokens
           (DENDRA uses a constant-size synaptic state, GPT uses a KV cache).
Randomly initialized models: speed does not depend on the weights.

Run:  python scripts/bench_inference.py [--device cpu|cuda] [--threads 2]
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dendra.model import DENDRA, GPT

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--threads", type=int, default=2)
p.add_argument("--ctx", default="128,512,1024,2048,4096")
p.add_argument("--decode", type=int, default=16)
p.add_argument("--out", default="experiments/results/bench_cpu.json")
a = p.parse_args()
torch.set_num_threads(a.threads)
dev = a.device
V, d, L, H, NE = 8192, 384, 6, 6, 576
CTX = [int(c) for c in a.ctx.split(",")]


def sync():
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


torch.manual_seed(0)
models = {
    "dendra": DENDRA(V, d=d, n_regions=L, n_heads=H, n_e=NE).to(dev).eval(),
    "gpt": GPT(V, d=d, n_layers=L, n_heads=H, ctx=max(CTX) + a.decode + 2).to(dev).eval(),
}
for k, m in models.items():
    print(f"{k}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")


def mem_mb(st):
    if "kv" in st:
        return sum(k.numel() + v.numel() for k, v in st["kv"]) * 2 / 1e6  # counted as fp16
    return sum(S.numel() for row in st["S"] for S in row) * 2 / 1e6


rows = []
with torch.no_grad():
    for n in CTX:
        x = torch.randint(0, V, (1, n), device=dev)
        for k, m in models.items():
            m.prefill(x[:, :32])  # warm-up
            sync(); t = time.perf_counter(); lg, st = m.prefill(x); sync(); pre = time.perf_counter() - t
            tok = lg[:, -1].argmax(-1, keepdim=True)
            m.step_token(tok, st)
            sync(); t = time.perf_counter()
            for _ in range(a.decode):
                lg, st = m.step_token(tok, st)
            sync(); dec = (time.perf_counter() - t) / a.decode
            r = dict(model=k, ctx=n, prefill_ms=pre * 1e3, decode_ms=dec * 1e3, state_mb=mem_mb(st))
            rows.append(r)
            print(f"{k:6} context {n:5d} | prefill {r['prefill_ms']:8.1f} ms | decode {r['decode_ms']:6.2f} ms/token | "
                  f"memory {r['state_mb']:6.2f} MB", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"device": dev, "threads": a.threads, "rows": rows}, open(a.out, "w"), indent=1)
