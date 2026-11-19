"""
DENDRA: a language model built from biological neuron mechanisms.

Main components
  LIF neuron + refractory period -> DendriticSpikingLayer (0/1 spikes, soft reset, adaptive threshold)
  Nonlinear dendrites            -> each neuron has B branches, each with its own local nonlinearity
  E (80%) / I (20%) populations  -> Dale's law: E->I and I->E weights are positive; I subtracts from E (soft k-WTA)
  Hebb + STDP + NMDA             -> HebbianSTDP: S <- lam*S + w * v k^T  (replaces attention)
  Neuromodulators (NE, ACh)      -> Neuromodulators: global gain and memory-write gate
  Feedback connections (L6)      -> several up/down passes, higher regions send signals down
  Spinal reflex                  -> early-exit head after region 1
  Homeostasis                    -> loss keeping the firing rate near a target (in the training script)

Three computation modes for the same weights (outputs match, max error < 1e-5):
  forward()      parallel over time, used for training
  prefill()      reads a prompt chunk by chunk, O(n * C)
  step_token()   generates one token with a constant-size state, O(1) memory per token
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- spikes + surrogate gradient
class _Spike(torch.autograd.Function):
    """All-or-none: spike = 1 when the membrane potential crosses threshold. Surrogate gradient: fast sigmoid."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + ctx.alpha * x.abs()) ** 2, None


def spike(x, alpha=2.0):
    return _Spike.apply(x, alpha)


class DivisiveNorm(nn.Module):
    """Divisive normalization by an interneuron pool: divide by the mean activity energy."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


# ---------------------------------------------------------------- L2/3: Hebbian/STDP synapses
class HebbianSTDP(nn.Module):
    """
    Fast-weight memory that replaces attention.
      S_t = lam * S_{t-1} + w_t * v_t k_t^T        (Hebb: pre and post active together -> stronger synapse)
      o_t = S_t q_t
    - lam < 1: synaptic decay; each head has its own time constant (4..256 tokens), learned.
    - Parallel form: o = ((Q K^T) * D * w) V with D_ij = lam^(i-j) for j <= i. This is a causal STDP window
      (pre-before-post strengthens, decaying with distance), so no position embedding is needed.
    - w_t = NMDA(x_t) * ACh_t: write only when the postsynaptic neuron is depolarized (NMDA gate) and ACh is high.
    """

    def __init__(self, d, n_heads, tau_min=4.0, tau_max=256.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.nmda = nn.Linear(d, n_heads)  # coincidence-detector gate
        self.og = nn.Linear(d, d, bias=False)  # output gate (postsynaptic depolarization)
        self.out = nn.Linear(d, d, bias=False)
        self.onorm = DivisiveNorm(self.dh)
        tau = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), n_heads))
        lam = 1.0 - 1.0 / tau
        self.lam_logit = nn.Parameter(torch.log(lam / (1 - lam)))  # learnable
        nn.init.constant_(self.nmda.bias, 2.0)

    def lam(self):
        return torch.sigmoid(self.lam_logit)  # [h]

    def _proj(self, x, ach):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)  # [B,T,h,dh]
        q = q * self.dh ** -0.5
        k = F.normalize(k, dim=-1)
        w = torch.sigmoid(self.nmda(x))  # [B,T,h]
        if ach is not None:
            w = w * ach
        return q, k, v, w

    def forward(self, x, ach=None):
        B, T, _ = x.shape
        q, k, v, w = self._proj(x, ach)
        idx = torch.arange(T, device=x.device)
        dist = (idx[:, None] - idx[None, :]).float()  # i - j
        causal = dist >= 0
        loglam = torch.log(self.lam())[:, None, None]  # [h,1,1]
        D = torch.where(causal, torch.exp(loglam * dist.clamp(min=0)), torch.zeros((), device=x.device))
        A = torch.einsum("bihd,bjhd->bhij", q, k) * D  # [B,h,T,T]
        A = A * w.permute(0, 2, 1)[:, :, None, :]  # write strength of each token j
        o = torch.einsum("bhij,bjhd->bihd", A, v)
        return self._finish(x, o)

    def _finish(self, x, o):
        B, T = x.shape[:2]
        o = self.onorm(o).reshape(B, T, -1)
        return self.out(o * F.silu(self.og(x)))

    def chunked(self, x, ach, S, C=128):
        """Read a sequence in chunks of C tokens: parallel inside a chunk, state S carried between chunks.
        Same result as forward() but O(n*C) instead of O(n^2); also returns the final state."""
        B, T, _ = x.shape
        q, k, v, w = self._proj(x, ach)
        lam = self.lam()  # [h]
        loglam = torch.log(lam)
        outs = []
        for s0 in range(0, T, C):
            qc, kc, vc, wc = q[:, s0:s0 + C], k[:, s0:s0 + C], v[:, s0:s0 + C], w[:, s0:s0 + C]
            n = qc.shape[1]
            idx = torch.arange(n, device=x.device)
            dist = (idx[:, None] - idx[None, :]).float()
            D = torch.where(dist >= 0, torch.exp(loglam[:, None, None] * dist.clamp(min=0)), torch.zeros((), device=x.device))
            A = torch.einsum("bihd,bjhd->bhij", qc, kc) * D * wc.permute(0, 2, 1)[:, :, None, :]
            o = torch.einsum("bhij,bjhd->bihd", A, vc)  # within-chunk part
            decay_in = torch.exp(loglam[None, :] * (idx[:, None] + 1).float())  # lam^(a+1) [n,h]
            o = o + torch.einsum("bhde,bihe->bihd", S.to(qc.dtype), qc) * decay_in[None, :, :, None].to(qc.dtype)
            decay_out = torch.exp(loglam[None, :] * (n - 1 - idx)[:, None].float())  # lam^(n-1-b) [n,h]
            wv = vc * (wc * decay_out[None].to(wc.dtype))[..., None]
            S = lam[None, :, None, None] ** n * S + torch.einsum("bjhd,bjhe->bhde", wv.float(), kc.float())
            outs.append(o)
        return self._finish(x, torch.cat(outs, 1)), S

    def step(self, x, ach, S):
        """One-token inference: constant-size memory S [B,h,dh,dh], no KV cache."""
        q, k, v, w = self._proj(x, ach)  # x [B,1,d]
        q, k, v, w = q[:, 0], k[:, 0], v[:, 0], w[:, 0]
        lam = self.lam()[None, :, None, None]
        S = lam * S + w[..., None, None] * torch.einsum("bhd,bhe->bhde", v, k)
        o = torch.einsum("bhde,bhe->bhd", S, q)[:, None]
        return self._finish(x, o), S


# ---------------------------------------------------------------- L5: spiking dendritic neurons
class DendriticSpikingLayer(nn.Module):
    """
    E population with dendrites + inhibitory I population, run for T gamma ticks per token.
      branch input    c_b = W_b x                (synapses on each dendritic branch)
      soma input      I = sum_b a_b * g(c_b)     (g: local branch nonlinearity, a_b >= 0)
      E potential     u <- beta*u + gain*I - W_IE s_I
      E spikes        s_E = H(u - (theta + adapt)); soft reset u -= theta*s_E; adapt <- rho*adapt + gamma*s_E
      I population    ui <- beta_i*ui + W_EI s_E;  s_I = H(ui - theta_I)
    Dale's law: W_EI, W_IE >= 0 (softplus). Output = E firing rate -> read out into the residual stream.
    """

    def __init__(self, d, n_e, branches=4, T=4, inh_ratio=0.25, theta=1.0, spike_in=False):
        super().__init__()
        self.spike_in = spike_in
        if spike_in:  # encode the input as ON/OFF spikes (like retinal cells) -> dendritic synapses only add
            self.theta_in_raw = nn.Parameter(torch.full((d,), 1.0))
        self.ne, self.B, self.T = n_e, branches, T
        self.ni = max(1, int(n_e * inh_ratio))  # inhibitory neurons are 20% of the total
        self.dend = nn.Linear(d, n_e * branches)
        self.branch_raw = nn.Parameter(torch.zeros(n_e, branches))  # softplus(0)=0.69
        self.beta_raw = nn.Parameter(torch.full((n_e,), 1.5))  # membrane leak, sigmoid ~ 0.82
        self.w_ei_raw = nn.Parameter(torch.randn(n_e, self.ni) * 0.1 - 4.0)
        self.w_ie_raw = nn.Parameter(torch.randn(self.ni, n_e) * 0.1 - 3.0)
        self.theta, self.theta_i = theta, 1.0
        self.rho, self.gamma = 0.5, 0.5  # refractory period / threshold adaptation
        self.out = nn.Linear(n_e, d, bias=False)
        nn.init.normal_(self.out.weight, std=0.02)

    def encode(self, x):
        th = F.softplus(self.theta_in_raw) + 0.1
        up, un = torch.zeros_like(x), torch.zeros_like(x)
        cp, cn = torch.zeros_like(x), torch.zeros_like(x)
        for _ in range(self.T):
            up = up + x
            un = un - x
            sp, sn = spike(up - th), spike(un - th)
            up, un = up - th * sp, un - th * sn
            cp, cn = cp + sp, cn + sn
        self.last_in_spikes = (cp + cn).sum(-1).mean().detach()  # input spikes per token
        return (cp - cn) / self.T * th  # quantized value, same scale

    def forward(self, x, gain=None):
        Bt = x.shape[:-1]
        if self.spike_in:
            x = self.encode(x)
        c = self.dend(x).view(*Bt, self.ne, self.B)
        I = (F.softplus(self.branch_raw) * F.silu(c)).sum(-1)  # spatial integration over dendrites
        if gain is not None:
            I = I * gain
        beta = torch.sigmoid(self.beta_raw)
        w_ei = F.softplus(self.w_ei_raw)
        w_ie = F.softplus(self.w_ie_raw)
        u = torch.zeros_like(I)
        adapt = torch.zeros_like(I)
        ui = torch.zeros(*Bt, self.ni, device=x.device, dtype=x.dtype)
        s_i = torch.zeros_like(ui)
        count = torch.zeros_like(I)
        count_i = torch.zeros_like(ui)
        for _ in range(self.T):  # T gamma ticks
            u = beta * u + I - s_i @ w_ie
            s = spike(u - (self.theta + adapt))
            u = u - self.theta * s
            adapt = self.rho * adapt + self.gamma * s
            ui = 0.5 * ui + s @ w_ei
            s_i = spike(ui - self.theta_i)
            ui = ui - self.theta_i * s_i
            count = count + s
            count_i = count_i + s_i
        rate = count / self.T
        self.last_rate = rate.mean()
        self.last_rate_i = (count_i / self.T).mean().detach()
        return self.out(rate)


# ---------------------------------------------------------------- one cortical region (6-layer column)
class CorticalRegion(nn.Module):
    """L4 input (divisive inhibition) -> L2/3 mixing over time -> L5 spiking output -> L6 receives feedback."""

    def __init__(self, d, n_heads, n_e, branches, T, spike_in=False):
        super().__init__()
        self.l4a = DivisiveNorm(d)
        self.l23 = HebbianSTDP(d, n_heads)
        self.l4b = DivisiveNorm(d)
        self.l5 = DendriticSpikingLayer(d, n_e, branches, T, spike_in=spike_in)
        self.l6_norm = DivisiveNorm(d)
        self.l6 = nn.Linear(d, d, bias=False)  # receives predictions from the region above
        nn.init.zeros_(self.l6.weight)

    def topdown(self, h_above):
        return self.l6(self.l6_norm(h_above))

    def forward(self, h, gain=None, ach=None, fb=None):
        if fb is not None:
            h = h + fb
        h = h + self.l23(self.l4a(h), ach)
        h = h + self.l5(self.l4b(h), gain)
        return h

    def forward_chunked(self, h, gain, ach, fb, S, C=128):
        if fb is not None:
            h = h + fb
        o, S = self.l23.chunked(self.l4a(h), ach, S, C)
        h = h + o
        h = h + self.l5(self.l4b(h), gain)
        return h, S

    def step(self, h, gain, ach, fb, S):
        if fb is not None:
            h = h + fb
        o, S = self.l23.step(self.l4a(h), ach, S)
        h = h + o
        h = h + self.l5(self.l4b(h), gain)
        return h, S


class Neuromodulators(nn.Module):
    """'Brainstem': from a baseline level (causal EMA of the input) emits noradrenaline (gain) and acetylcholine (write gate)."""

    def __init__(self, d, n_regions, n_heads, decay=0.9):
        super().__init__()
        self.L, self.h, self.decay = n_regions, n_heads, decay
        self.norm = DivisiveNorm(d)
        self.proj = nn.Linear(d, n_regions * (1 + n_heads))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)  # start: gain = 1, ach = 1

    def _split(self, m):
        m = 2 * torch.sigmoid(self.proj(self.norm(m)))
        m = m.view(*m.shape[:-1], self.L, 1 + self.h)
        return m[..., 0:1], m[..., 1:]  # gain [..,L,1], ach [..,L,h]

    def forward(self, x, K=256):
        # causal EMA as a 1-D convolution (kernel truncated at K steps: decay^K ~ 1e-12), O(n*K) instead of O(n^2)
        B, T, d = x.shape
        K = min(K, T)
        ker = (self.decay ** torch.arange(K - 1, -1, -1, device=x.device, dtype=torch.float32)).to(x.dtype)
        xt = F.pad(x.transpose(1, 2), (K - 1, 0))
        ema = F.conv1d(xt, ker.view(1, 1, K).expand(d, 1, K).contiguous(), groups=d).transpose(1, 2)
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        norm = (1 - self.decay ** (t + 1).clamp(max=K)) / (1 - self.decay)
        return self._split(ema / norm[None, :, None].to(ema.dtype))

    def step(self, x, m_state, n_seen):
        # normalized EMA, identical to the parallel form
        m_state = self.decay * m_state + x
        norm = (1 - self.decay ** (n_seen + 1)) / (1 - self.decay)
        return self._split(m_state / norm), m_state


class DENDRA(nn.Module):
    def __init__(self, vocab, d=128, n_regions=4, n_heads=4, n_e=192, branches=4, T=4,
                 fb_cycles=1, use_neuromod=True, reflex=True, spike_in=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)  # sensory encoding
        nn.init.normal_(self.emb.weight, std=0.02)
        self.regions = nn.ModuleList([CorticalRegion(d, n_heads, n_e, branches, T, spike_in) for _ in range(n_regions)])
        self.nm = Neuromodulators(d, n_regions, n_heads) if use_neuromod else None
        self.fb_cycles = fb_cycles
        self.norm_out = DivisiveNorm(d)
        self.reflex = reflex
        if reflex:
            self.norm_reflex = DivisiveNorm(d)
        self.n_heads = n_heads

    def _head(self, h, norm):
        return norm(h) @ self.emb.weight.t()

    def forward(self, idx):
        x = self.emb(idx)
        gain = ach = None
        if self.nm is not None:
            gain, ach = self.nm(x)
        prev = None
        for c in range(self.fb_cycles):
            h, outs = x, []
            for l, reg in enumerate(self.regions):
                fb = None
                if prev is not None and l + 1 < len(self.regions):
                    fb = reg.topdown(prev[l + 1])  # prediction from the region above in the previous pass
                h = reg(h, None if gain is None else gain[:, :, l], None if ach is None else ach[:, :, l], fb)
                outs.append(h)
            prev = outs
        logits = self._head(h, self.norm_out)
        reflex_logits = self._head(prev[0], self.norm_reflex) if self.reflex else None
        return logits, reflex_logits

    def spike_rates(self):
        return [r.l5.last_rate for r in self.regions]

    def init_state(self, B, dev):
        d = self.emb.weight.shape[1]
        L, h = len(self.regions), self.n_heads
        dh = d // h
        return {"S": [[torch.zeros(B, h, dh, dh, device=dev) for _ in range(L)] for _ in range(self.fb_cycles)],
                "m": torch.zeros(B, 1, d, device=dev), "t": 0}

    @torch.no_grad()
    def prefill(self, idx, C=128):
        """Read the whole prompt once (in chunks); return the last-token logits + state for further generation."""
        B, T = idx.shape
        st = self.init_state(B, idx.device)
        x = self.emb(idx)
        gain = ach = None
        if self.nm is not None:
            gain, ach = self.nm(x)
            m = torch.zeros(B, 1, x.shape[-1], device=x.device, dtype=torch.float32)
            dec = self.nm.decay ** torch.arange(T - 1, -1, -1, device=x.device, dtype=torch.float32)
            st["m"] = (x.float() * dec[None, :, None]).sum(1, keepdim=True)  # unnormalized EMA sum
        st["t"] = T
        L = len(self.regions)
        prev = None
        for c in range(self.fb_cycles):
            h, outs = x, []
            for l, reg in enumerate(self.regions):
                fb = reg.topdown(prev[l + 1]) if (prev is not None and l + 1 < L) else None
                h, st["S"][c][l] = reg.forward_chunked(h, None if gain is None else gain[:, :, l],
                                                       None if ach is None else ach[:, :, l], fb, st["S"][c][l], C)
                outs.append(h)
            prev = outs
        return self._head(h[:, -1:], self.norm_out), st

    @torch.no_grad()
    def step_token(self, tok, st):
        """Generate 1 token using only the constant-size synaptic state (O(1) memory per token)."""
        L = len(self.regions)
        x = self.emb(tok)
        gain = ach = None
        if self.nm is not None:
            (gain, ach), st["m"] = self.nm.step(x, st["m"], st["t"])
        st["t"] += 1
        prev = None
        for c in range(self.fb_cycles):
            hcur, outs = x, []
            for l, reg in enumerate(self.regions):
                fb = reg.topdown(prev[l + 1]) if (prev is not None and l + 1 < L) else None
                hcur, st["S"][c][l] = reg.step(hcur, None if gain is None else gain[:, :, l],
                                               None if ach is None else ach[:, :, l], fb, st["S"][c][l])
                outs.append(hcur)
            prev = outs
        return self._head(hcur, self.norm_out), st

    @torch.no_grad()
    def forward_recurrent(self, idx):
        """Run token by token with the constant-size state (to check it matches the parallel forward)."""
        st = self.init_state(idx.shape[0], idx.device)
        outs = []
        for t in range(idx.shape[1]):
            lg, st = self.step_token(idx[:, t:t + 1], st)
            outs.append(lg)
        return torch.cat(outs, 1)


# ---------------------------------------------------------------- GPT baseline (standard causal attention)
class GPTBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=384, n_layers=6, n_heads=6, ctx=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([GPTBlock(d, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.emb.weight.t(), None

    def _run(self, x, cache, causal):
        for i, b in enumerate(self.blocks):
            B, T, D = x.shape
            q, k, v = b.qkv(b.ln1(x)).view(B, T, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], 2); v = torch.cat([cache[i][1], v], 2)
            cache[i] = (k, v)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(B, T, D)
            x = x + b.proj(a)
            x = x + b.mlp(b.ln2(x))
        return self.ln(x) @ self.emb.weight.t(), cache

    @torch.no_grad()
    def prefill(self, idx):
        """Read the prompt and store the KV cache (memory grows with context length)."""
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        lg, cache = self._run(x, [None] * len(self.blocks), True)
        return lg[:, -1:], {"kv": cache, "t": idx.shape[1]}

    @torch.no_grad()
    def step_token(self, tok, st):
        pos = min(st["t"], self.pos.num_embeddings - 1)
        x = self.emb(tok) + self.pos(torch.tensor([pos], device=tok.device))
        lg, st["kv"] = self._run(x, st["kv"], False)
        st["t"] += 1
        return lg, st
