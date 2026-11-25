"""Model sizes used in the scaling sweep.

Each preset keeps head_dim = 64 and grows width and depth together. The
non-embedding parameter count is roughly 12 * n_layer * n_embd^2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 512
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, GPTConfig] = {
    "5m": GPTConfig(vocab_size=8192, n_layer=6, n_head=4, n_embd=256),
    "15m": GPTConfig(vocab_size=8192, n_layer=8, n_head=6, n_embd=384),
    "40m": GPTConfig(vocab_size=8192, n_layer=12, n_head=8, n_embd=512),
    "85m": GPTConfig(vocab_size=8192, n_layer=12, n_head=12, n_embd=768),
}
