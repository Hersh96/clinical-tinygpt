"""Text generation, with or without the KV cache.

Both paths follow the same rule for long outputs: the model only sees the
most recent tokens inside a window that starts at ``ctx_start``. When the
window is full (``block_size`` tokens), it jumps forward so the newest
``block_size // 2`` tokens are kept. With the cache, that jump means clearing
the cache and re-reading ("prefilling") those tokens. Because the rule is the
same, cached and uncached greedy generation produce identical tokens; the
cached one is just faster.
"""
from __future__ import annotations

import torch

from ctg.model.kv_cache import KVCache


def _pick(logits: torch.Tensor, temperature: float, top_k: int | None, gen: torch.Generator | None) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=gen)


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, use_cache: bool = True, seed: int | None = None) -> torch.Tensor:
    block = model.cfg.block_size
    gen = None
    if seed is not None:
        gen = torch.Generator(device=idx.device).manual_seed(seed)
    ctx_start = max(0, idx.shape[1] - block)
    cache = None
    if use_cache:
        p = next(model.parameters())
        cfg = model.cfg
        cache = KVCache(cfg.n_layer, idx.shape[0], cfg.n_head, cfg.n_embd // cfg.n_head, block, p.device, p.dtype)
    filled = 0  # positions of the current window already in the cache
    for _ in range(max_new_tokens):
        if idx.shape[1] - ctx_start > block:  # window full: slide it forward
            ctx_start = idx.shape[1] - block // 2
            filled = 0
            if cache is not None:
                cache.reset()
        window = idx[:, ctx_start:]
        if cache is None:
            logits, _ = model(window)
        else:
            new = window[:, filled:]
            logits, _ = model(new, cache=cache, start_pos=filled)
            filled = window.shape[1]
        nxt = _pick(logits[:, -1, :], temperature, top_k, gen)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
