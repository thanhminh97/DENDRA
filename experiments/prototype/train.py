"""Train DENDRA and a GPT baseline on Tiny Shakespeare (character level)."""
import argparse, json, math, time
import torch
import torch.nn.functional as F
from dendra_proto import DENDRA, GPT

p = argparse.ArgumentParser()
p.add_argument("--model", default="dendra", choices=["dendra", "gpt"])
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--ctx", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--fb", type=int, default=1)
p.add_argument("--no_nm", action="store_true")
p.add_argument("--spike_in", action="store_true")
p.add_argument("--rate_target", type=float, default=0.15)
p.add_argument("--homeo", type=float, default=1.0)
p.add_argument("--reflex_w", type=float, default=0.3)
p.add_argument("--data", default="data/input.txt")
p.add_argument("--out", default="result.json")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed)
torch.set_num_threads(int(__import__("os").environ.get("NT", "1")))

text = open(a.data, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train, val = data[:n], data[n:]
V = len(chars)


def batch(src, bs):
    ix = torch.randint(len(src) - a.ctx - 1, (bs,))
    x = torch.stack([src[i:i + a.ctx] for i in ix])
    y = torch.stack([src[i + 1:i + a.ctx + 1] for i in ix])
    return x, y


if a.model == "dendra":
    model = DENDRA(V, fb_cycles=a.fb, use_neuromod=not a.no_nm, spike_in=a.spike_in)
else:
    model = GPT(V, ctx=a.ctx)
n_params = sum(p.numel() for p in model.parameters())
print(f"{a.model}: {n_params/1e6:.3f}M params", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

log = []
t0 = time.time()
for step in range(a.steps):
    model.train()
    x, y = batch(train, a.bs)
    logits, rlog = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    total = loss
    if a.model == "dendra":
        rates = torch.stack(model.spike_rates())
        homeo = ((rates - a.rate_target) ** 2).sum()  # Na/K pump: keep activity at the target level
        total = total + a.homeo * homeo
        if rlog is not None:
            total = total + a.reflex_w * F.cross_entropy(rlog.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 100 == 0 or step == a.steps - 1:
        msg = f"step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)"
        if a.model == "dendra":
            msg += " rates " + " ".join(f"{r.item():.3f}" for r in rates)
        print(msg, flush=True)
        log.append({"step": step, "loss": loss.item()})
train_time = time.time() - t0


@torch.no_grad()
def evaluate(n_batches=40):
    model.eval()
    g = torch.Generator().manual_seed(123)
    tot, tot_r, cnt = 0.0, 0.0, 0
    conf_all, okr_all, okf_all = [], [], []
    rates = []
    for _ in range(n_batches):
        ix = torch.randint(len(val) - a.ctx - 1, (16,), generator=g)
        x = torch.stack([val[i:i + a.ctx] for i in ix])
        y = torch.stack([val[i + 1:i + a.ctx + 1] for i in ix])
        logits, rlog = model(x)
        tot += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum").item()
        cnt += y.numel()
        okf_all.append((logits.argmax(-1) == y).flatten())
        if rlog is not None:
            tot_r += F.cross_entropy(rlog.reshape(-1, V), y.reshape(-1), reduction="sum").item()
            pr = rlog.softmax(-1)
            conf_all.append(pr.max(-1).values.flatten())
            okr_all.append((pr.argmax(-1) == y).flatten())
        if a.model == "dendra":
            rates.append(torch.stack(model.spike_rates()))
    res = {"val_loss": tot / cnt, "val_bpc": tot / cnt / math.log(2),
           "acc_full": torch.cat(okf_all).float().mean().item()}
    if conf_all:
        conf, okr, okf = torch.cat(conf_all), torch.cat(okr_all), torch.cat(okf_all)
        res["reflex_bpc"] = tot_r / cnt / math.log(2)
        res["reflex"] = []
        for thr in [0.5, 0.7, 0.9]:
            ex = conf > thr
            mixed = torch.where(ex, okr, okf).float().mean().item()
            res["reflex"].append({"thr": thr, "exit_frac": ex.float().mean().item(),
                                  "acc_exited": okr[ex].float().mean().item() if ex.any() else None,
                                  "acc_mixed": mixed})
    if rates:
        res["spike_rates"] = torch.stack(rates).mean(0).tolist()
    return res


res = evaluate()
res.update({"model": a.model, "params": n_params, "steps": a.steps, "train_time_s": train_time,
            "fb": a.fb, "neuromod": not a.no_nm, "spike_in": a.spike_in, "log": log})

if a.model == "dendra":
    # check: parallel mode (training) == token-by-token mode (O(1) inference)
    model.eval()
    x, _ = batch(val, 2)
    x = x[:, :64]
    with torch.no_grad():
        lp, _ = model(x)
        lr_ = model.forward_recurrent(x)
    res["parallel_vs_recurrent_maxdiff"] = (lp - lr_).abs().max().item()
    # E/I: inhibitory population firing rate
    res["inh_rates"] = [r.l5.last_rate_i.item() for r in model.regions]
    res["lambda_tau"] = [(1 / (1 - r.l23.lam())).tolist() for r in model.regions]

# text generation
model.eval()
ctx = torch.tensor([[stoi[c] for c in "ROMEO:\n"]])
with torch.no_grad():
    for _ in range(300):
        logits, _ = model(ctx[:, -a.ctx:])
        nxt = torch.multinomial((logits[0, -1] / 0.8).softmax(-1), 1)
        ctx = torch.cat([ctx, nxt[None]], 1)
res["sample"] = "".join(chars[i] for i in ctx[0].tolist())
print(json.dumps({k: v for k, v in res.items() if k not in ("log",)}, ensure_ascii=False, indent=1))
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
torch.save(model.state_dict(), a.out.replace(".json", ".pt"))
