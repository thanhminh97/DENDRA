"""Fast decoder for DENDRA and GPT: static state + CUDA Graph (+ optional torch.compile).

    from dendra.fast import make_decoder
    logits, dec = make_decoder(model, prompt_ids)          # prefill + capture a CUDA Graph (if a GPU is present)
    next_logits = dec(next_token_ids)                       # each call = one generation step

Why: one DENDRA generation step launches hundreds of tiny kernels (4 ticks x E/I microcircuit x regions),
so on a GPU the CPU-side launch overhead dominates (~20 ms/step regardless of context length). Fixes:
  1) Static state tensors -> the whole step is captured once as a CUDA Graph and replayed with one launch.
  2) q, k, v + NMDA gate + output gate fused into one matmul; softplus/sigmoid of parameters precomputed.
  3) Optional torch.compile to fuse the element-wise chain of the LIF loop into fewer kernels.
GPT gets the same treatment (preallocated KV cache + CUDA Graph) for a fair comparison.
"""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import GPT

AMP_DTYPE = torch.float16 if torch.cuda.is_available() else None


def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE else contextlib.nullcontext()


class FastDendra(nn.Module):
    """Inference version of DENDRA: same weights, same outputs, the state is a fixed-size tensor."""

    def __init__(self, m):
        super().__init__()
        assert not any(r.l5.spike_in for r in m.regions), "FastDendra does not support spike_in yet"
        self.m = m.eval()
        self.L, self.h = len(m.regions), m.n_heads
        self.d = m.emb.weight.shape[1]
        self.dh = self.d // self.h
        self.C = m.fb_cycles
        with torch.no_grad():
            self.Wp, self.bp, self.lam, self.branch, self.beta, self.wei, self.wie = [], [], [], [], [], [], []
            for r in m.regions:
                a = r.l23
                W = torch.cat([a.qkv.weight, a.nmda.weight, a.og.weight], 0)        # 1 matmul instead of 3
                b = torch.cat([torch.zeros(3 * self.d, device=W.device), a.nmda.bias,
                               torch.zeros(self.d, device=W.device)])
                self.Wp.append(W); self.bp.append(b)
                self.lam.append(a.lam().view(1, self.h, 1, 1))
                self.branch.append(F.softplus(r.l5.branch_raw))
                self.beta.append(torch.sigmoid(r.l5.beta_raw))
                self.wei.append(F.softplus(r.l5.w_ei_raw))
                self.wie.append(F.softplus(r.l5.w_ie_raw))

    def init_state(self, B, dev):
        S = torch.zeros(self.C, self.L, B, self.h, self.dh, self.dh, device=dev)
        return S, torch.zeros(B, self.d, device=dev), torch.zeros((), device=dev)

    @torch.no_grad()
    def from_prefill(self, st):
        S = torch.stack([torch.stack(row) for row in st["S"]]).float()
        return S, st["m"][:, 0].float(), torch.tensor(float(st["t"]), device=S.device)

    def _region(self, l, h, gain, ach, S):
        r, d, H, dh = self.m.regions[l], self.d, self.h, self.dh
        x = r.l4a(h)
        P = F.linear(x, self.Wp[l], self.bp[l])
        q = P[:, :d].view(-1, H, dh) * dh ** -0.5
        k = F.normalize(P[:, d:2 * d].view(-1, H, dh), dim=-1)
        v = P[:, 2 * d:3 * d].view(-1, H, dh)
        w = torch.sigmoid(P[:, 3 * d:3 * d + H])
        if ach is not None:
            w = w * ach
        og = P[:, 3 * d + H:]
        S = self.lam[l] * S + (w[..., None, None] * v[..., :, None] * k[..., None, :]).to(S.dtype)  # Hebb
        o = torch.einsum("bhde,bhe->bhd", S.to(q.dtype), q)
        o = r.l23.onorm(o).reshape(-1, d)
        h = h + r.l23.out(o * F.silu(og))
        l5 = r.l5
        c = l5.dend(r.l4b(h)).view(-1, l5.ne, l5.B)
        I = (self.branch[l] * F.silu(c)).sum(-1)
        if gain is not None:
            I = I * gain
        u = torch.zeros_like(I); ad = torch.zeros_like(I); cnt = torch.zeros_like(I)
        ui = torch.zeros(I.shape[0], l5.ni, device=I.device, dtype=I.dtype); si = torch.zeros_like(ui)
        beta, wie, wei = self.beta[l].to(I.dtype), self.wie[l].to(I.dtype), self.wei[l].to(I.dtype)
        for _ in range(l5.T):                                   # T gamma ticks, E/I microcircuit
            u = beta * u + I - si @ wie
            s = (u > l5.theta + ad).to(I.dtype)
            u = u - l5.theta * s
            ad = l5.rho * ad + l5.gamma * s
            ui = 0.5 * ui + s @ wei
            si = (ui > l5.theta_i).to(I.dtype)
            ui = ui - l5.theta_i * si
            cnt = cnt + s
        return h + l5.out(cnt / l5.T), S

    def step(self, tok, S, mstate, t):
        """tok [B]; S [C,L,B,h,dh,dh]; mstate [B,d]; t = number of tokens read so far (tensor).
        Updates S, mstate, t IN PLACE (so the CUDA Graph can reuse the same addresses) and returns logits."""
        m = self.m
        x = m.emb(tok)
        gain = ach = None
        if m.nm is not None:
            mnew = m.nm.decay * mstate + x.float()
            norm = (1 - torch.pow(torch.full((), m.nm.decay, device=x.device), t + 1)) / (1 - m.nm.decay)
            z = 2 * torch.sigmoid(m.nm.proj(m.nm.norm((mnew / norm).to(x.dtype))))
            z = z.view(-1, self.L, 1 + self.h)
            gain, ach = z[..., :1], z[..., 1:]
        newS, prev = [], None
        for c in range(self.C):
            hcur, outs, row = x, [], []
            for l in range(self.L):
                if prev is not None and l + 1 < self.L:
                    hcur = hcur + m.regions[l].topdown(prev[l + 1])
                hcur, Sl = self._region(l, hcur, None if gain is None else gain[:, l],
                                        None if ach is None else ach[:, l], S[c, l])
                outs.append(hcur); row.append(Sl)
            prev = outs
            newS.append(torch.stack(row))
        logits = m.norm_out(hcur) @ m.emb.weight.t()
        S.copy_(torch.stack(newS))
        if m.nm is not None:
            mstate.copy_(mnew)
        t.add_(1)
        return logits


class FastGPT(nn.Module):
    """GPT with a preallocated KV cache (no torch.cat per step) so it can run inside a CUDA Graph."""

    def __init__(self, g, maxlen):
        super().__init__()
        self.g, self.maxlen = g.eval(), maxlen
        self.L, self.H = len(g.blocks), g.blocks[0].h
        self.d = g.emb.weight.shape[1]

    def init_state(self, B, dev, dtype=torch.float32):
        return torch.zeros(self.L, 2, B, self.H, self.maxlen, self.d // self.H, device=dev, dtype=dtype)

    @torch.no_grad()
    def from_prefill(self, st, dtype=torch.float32):
        kv = st["kv"]; n = kv[0][0].shape[2]; B = kv[0][0].shape[0]
        cache = self.init_state(B, kv[0][0].device, dtype)
        for i, (k, v) in enumerate(kv):
            cache[i, 0, :, :, :n] = k; cache[i, 1, :, :, :n] = v
        return cache, torch.tensor(n, device=cache.device)

    def step(self, tok, cache, pos):
        g = self.g
        x = g.emb(tok) + g.pos(pos.clamp(max=g.pos.num_embeddings - 1).view(1))      # [B, D]
        invalid = (torch.arange(self.maxlen, device=x.device) > pos).view(1, 1, -1)  # positions not written yet
        for i, b in enumerate(g.blocks):
            B, D = x.shape
            H, dh = b.h, D // b.h
            qkv = b.qkv(b.ln1(x)).view(B, 3, H, dh)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                                # [B, H, dh]
            cache[i, 0].index_copy_(2, pos.view(1), k[:, :, None].to(cache.dtype))   # write K/V in place
            cache[i, 1].index_copy_(2, pos.view(1), v[:, :, None].to(cache.dtype))
            att = torch.einsum("bhd,bhnd->bhn", q, cache[i, 0].to(q.dtype)).float() * dh ** -0.5
            att = att.masked_fill(invalid, float("-inf")).softmax(-1).to(q.dtype)
            a = torch.einsum("bhn,bhnd->bhd", att, cache[i, 1].to(q.dtype)).reshape(B, D)
            x = x + b.proj(a)
            x = x + b.mlp(b.ln2(x))
        pos.add_(1)
        return g.ln(x) @ g.emb.weight.t()


class Graphed:
    """Capture one generation step as a CUDA Graph and replay it with a single launch. Without a GPU it runs eagerly.
    fn(tok, *state) must update the state in place and return the logits."""

    def __init__(self, fn, tok, state, use_graph=True, use_compile=False):
        self.fn, self.state, self.graph = fn, state, None
        if use_compile:
            try:
                self.fn = torch.compile(fn, mode="max-autotune-no-cudagraphs", dynamic=False)
            except Exception as e:
                print("torch.compile failed, skipping:", type(e).__name__, str(e)[:120])
        if use_graph and tok.device.type == "cuda":
            self.tok = tok.clone()
            backup = [x.clone() for x in state]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad(), autocast():
                for _ in range(3):                      # warm-up (and compilation if compile is on)
                    self.fn(self.tok, *state)
            torch.cuda.current_stream().wait_stream(s)
            try:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph), torch.no_grad(), autocast():
                    self.out = self.fn(self.tok, *state)
            except Exception as e:                     # cannot capture -> run eagerly instead of crashing
                self.graph = None
                print(f"  (CUDA Graph unavailable for {type(fn.__self__).__name__}: "
                      f"{type(e).__name__}; running eagerly)")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            for x, b in zip(state, backup):             # restore the state to right after prefill
                x.copy_(b)

    def __call__(self, tok):
        if self.graph is None:
            with torch.no_grad(), autocast():
                return self.fn(tok, *self.state)
        self.tok.copy_(tok)
        self.graph.replay()
        return self.out


@torch.no_grad()
def make_decoder(model, prompt_ids, use_graph=True, use_compile=False, maxlen=None):
    """Read the prompt (prefill), then return (last-token logits, dec) where next_logits = dec(tok [B])."""
    with autocast():
        lg, st = model.prefill(prompt_ids)
    if isinstance(model, GPT):
        fm = FastGPT(model, maxlen or (prompt_ids.shape[1] + 256))
        state = list(fm.from_prefill(st, dtype=torch.float16 if AMP_DTYPE else torch.float32))
    else:
        fm = FastDendra(model)
        state = list(fm.from_prefill(st))
    tok0 = lg[:, -1].argmax(-1)
    runner = Graphed(fm.step, tok0, state, use_graph, use_compile)
    return lg[:, -1], runner
