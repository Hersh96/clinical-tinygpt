"""The KV cache: why generation doesn't have to redo all the work.

When a transformer generates text one token at a time, each new token needs
to attend to every earlier token. Attention uses a *key* and a *value* vector
for every earlier position. Those vectors never change once computed, so
instead of recomputing them for the whole sequence at every step (cost grows
with the square of the length), we store them here and only compute the key
and value for the newest token (cost grows linearly).

The trade-off is memory: 2 (k and v) x layers x heads x head_dim x length
numbers per sequence.
"""
from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layer: int, batch: int, n_head: int, head_dim: int, max_len: int, device, dtype):
        shape = (batch, n_head, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.max_len = max_len
        self.length = 0  # number of positions filled

    def update(self, layer: int, start_pos: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k/v for positions [start_pos, start_pos+T) and return all keys/values so far."""
        end = start_pos + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, have {self.max_len}")
        self.k[layer][:, :, start_pos:end] = k
        self.v[layer][:, :, start_pos:end] = v
        if layer == len(self.k) - 1:
            self.length = end
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def reset(self) -> None:
        self.length = 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)
