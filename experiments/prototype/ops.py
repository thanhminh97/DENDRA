"""Count operations per token (MAC = multiply-accumulate, AC = accumulate only when a spike arrives) and estimate energy."""
import json, sys
import torch
from dendra_proto import DENDRA

torch.set_num_threads(1)
E_MAC, E_AC = 4.6, 0.9  # pJ, 32-bit float, 45 nm (Horowitz 2014) - the usual convention in SNN papers
text = open("data/input.txt", encoding="utf-8").read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); val = data[int(0.9 * len(data)):]
V, d, h, ne, B, T, L, ctx = len(chars), 128, 4, 192, 4, 4, 4, 128
dh, ni = d // h, ne // 4


def gpt_ops(n_ctx=64):
    per_layer = 4 * d * d + 8 * d * d + 2 * n_ctx * d
    return {"mac": L * per_layer + d * V, "ac": 0}


def dendra_ops(run):
    r = json.load(open(f"runs/{run}.json"))
    m = DENDRA(V, fb_cycles=r["fb"], use_neuromod=r["neuromod"], spike_in=r.get("spike_in", False))
    m.load_state_dict(torch.load(f"runs/{run}.pt"))
    m.eval()
    g = torch.Generator().manual_seed(7)
    acc = {"e": torch.zeros(L), "i": torch.zeros(L), "in": torch.zeros(L)}
    nb = 8
    with torch.no_grad():
        for _ in range(nb):
            ix = torch.randint(len(val) - ctx, (16,), generator=g)
            m(torch.stack([val[i:i + ctx] for i in ix]))
            for l, reg in enumerate(m.regions):
                acc["e"][l] += reg.l5.last_rate / nb
                acc["i"][l] += reg.l5.last_rate_i / nb
                if reg.l5.spike_in:
                    acc["in"][l] += reg.l5.last_in_spikes / nb
    mac = ac = 0.0
    for l in range(L):
        mac += 5 * d * d + d * h + 2 * d * dh  # L2/3: q,k,v projections, gates, read/write of memory S
        e_sp = acc["e"][l].item() * ne * T  # E spikes per token
        i_sp = acc["i"][l].item() * ni * T
        if r.get("spike_in"):
            ac += acc["in"][l].item() * ne * B  # dendritic synapses receive spikes
        else:
            mac += d * ne * B
        mac += ne * B  # sum over branches
        ac += e_sp * ni + i_sp * ne + e_sp * d  # E->I, I->E, readout into the residual stream
    mac *= r["fb"]; ac *= r["fb"]
    if r["fb"] > 1:
        mac += (r["fb"] - 1) * (L - 1) * d * d
    if r["neuromod"]:
        mac += d * L * (1 + h)
    mac += d * V
    return {"mac": mac, "ac": ac, "e_rate": acc["e"].tolist(), "i_rate": acc["i"].tolist(),
            "in_spikes_per_token": acc["in"].tolist()}


out = {"gpt": gpt_ops()}
for run in sys.argv[1:]:
    out[run] = dendra_ops(run)
for k, v in out.items():
    v["energy_nJ"] = (v["mac"] * E_MAC + v["ac"] * E_AC) / 1000
    v["ac_share"] = v["ac"] / (v["mac"] + v["ac"])
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
json.dump(out, open("runs/ops.json", "w"), indent=1)
