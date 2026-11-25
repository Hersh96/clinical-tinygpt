"""A small GPT: token embeddings -> N transformer blocks -> next-token logits.

``forward`` returns logits of shape (batch, time, vocab): for every position,
a score for every possible next token. With ``targets`` it also returns the
cross-entropy loss, which is what training minimises. A random model scores
about ln(vocab_size) (e.g. 9.0 for 8,192 tokens) because it is guessing
uniformly; every point below that is something learned.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctg.model.config import GPTConfig
from ctg.model.layers import Block, RMSNorm, rope_tables


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: same table to read and write tokens
        cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scale residual outputs so deep stacks start stable
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None, cache=None, start_pos: int = 0, return_attn: bool = False):
        B, T = idx.shape
        if start_pos + T > self.cfg.block_size:
            raise ValueError(f"sequence of {start_pos + T} exceeds block_size {self.cfg.block_size}")
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        x = self.drop(self.tok_emb(idx))
        attns = []
        for i, block in enumerate(self.blocks):
            x, a = block(x, cos, sin, cache, i, start_pos, return_attn)
            attns.append(a)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_attn:
            return logits, loss, attns
        return logits, loss
