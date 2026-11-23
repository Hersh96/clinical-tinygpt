"""The building blocks of a modern decoder-only transformer.

A transformer *block* does two things to every token's vector:

1. **Attention** — each token looks back at earlier tokens and pulls in
   information from the relevant ones ("which earlier words matter for
   predicting the next one?").
2. **MLP** — each token's vector is transformed on its own ("think about what
   was gathered").

Each step is wrapped as ``x = x + f(norm(x))`` (a *residual* connection with
pre-normalisation), which keeps training stable as blocks are stacked.

Pieces used here (the same choices as Llama-style models):
- **RMSNorm** instead of LayerNorm: rescales a vector by its root-mean-square.
- **RoPE** (rotary position embeddings): position is encoded by rotating the
  query and key vectors by an angle that depends on the position, so
  attention scores depend on how far apart two tokens are.
- **SwiGLU** MLP: a gated feed-forward layer that trains better than ReLU/GELU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctg.model.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim/2) for rotary embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels of x (B, H, T, D) by position-dependent angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None, layer: int = 0, start_pos: int = 0, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, start_pos, k, v)   # now covers positions [0, start_pos+T)
        S = k.shape[2]
        # Query i sits at absolute position start_pos+i and may see keys 0..start_pos+i.
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        attn = None
        if return_attn:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            y = attn @ v
        elif cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), attn


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 64 * round(8 * cfg.n_embd / 3 / 64)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.n_embd), SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0, start_pos=0, return_attn=False):
        a, attn = self.attn(self.norm1(x), cos, sin, cache, layer, start_pos, return_attn)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, attn
