"""Learning-rate schedule: linear warmup, then cosine decay to a floor.

Warmup avoids huge, destabilising updates while the weights are still random;
cosine decay takes smaller steps late in training so the loss settles.
"""
from __future__ import annotations

import math


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float:
    min_lr = max_lr * min_lr_ratio
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
