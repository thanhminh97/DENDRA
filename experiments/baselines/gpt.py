"""
GPT baseline: a standard Transformer (pre-LayerNorm, causal self-attention), used only to compare with DENDRA.

    from experiments.baselines.gpt import GPT, make_gpt_decoder

GPT has the same interface as DENDRA:
  forward(idx)              parallel over the sequence, for training
  prefill(idx)              read a prompt and store the KV cache (grows with the context)
  step_token(tok, state)    generate one token from the KV cache
and a fast decoder (preallocated KV cache + CUDA Graph) built like DENDRA's, for a fair speed comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dendra.fast import AMP_DTYPE, Graphed, autocast


# ---------------------------------------------------------------- model
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


# ---------------------------------------------------------------- fast decoder
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


@torch.no_grad()
def make_gpt_decoder(model, prompt_ids, use_graph=True, use_compile=False, maxlen=None):
    """Read the prompt (prefill), then return (last-token logits, dec) where next_logits = dec(tok [B]).
    maxlen: largest context the preallocated KV cache must hold (default: prompt + 256)."""
    with autocast():
        lg, st = model.prefill(prompt_ids)
    fm = FastGPT(model, maxlen or (prompt_ids.shape[1] + 256))
    state = list(fm.from_prefill(st, dtype=torch.float16 if AMP_DTYPE else torch.float32))
    tok0 = lg[:, -1].argmax(-1)
    runner = Graphed(fm.step, tok0, state, use_graph, use_compile)
    return lg[:, -1], runner
