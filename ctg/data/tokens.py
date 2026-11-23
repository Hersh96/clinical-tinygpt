"""Turn the cleaned text into one long array of token ids, and sample batches.

Training a language model is next-token prediction: given tokens
``t[0..T-1]``, predict ``t[1..T]``. So a batch is just random windows from one
long token array, where the target ``y`` is the input ``x`` shifted by one.

Token ids are stored as ``uint16`` (2 bytes each, max id 65,535), which keeps
~100M tokens in ~200 MB and lets ``numpy.memmap`` read windows straight from
disk without loading the whole file into RAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctg.tokenizer.bpe import EOT, BPETokenizer


def encode_split(jsonl: Path, tok: BPETokenizer, out: Path) -> int:
    """Encode every document in ``jsonl`` followed by <|endoftext|>; return token count."""
    if max(tok.vocab) > np.iinfo(np.uint16).max or tok.special.get(EOT, 0) > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary does not fit in uint16")
    eot = tok.special[EOT]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out, "wb") as f:
        buf: list[int] = []
        for line in open(jsonl, encoding="utf-8"):
            buf.extend(tok.encode(json.loads(line)["text"]))
            buf.append(eot)
            if len(buf) > 1_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                buf = []
        np.asarray(buf, dtype=np.uint16).tofile(f)
        total += len(buf)
    return total


class TokenStream:
    """Random fixed-length windows from a uint16 token file."""

    def __init__(self, path: Path, block_size: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size:
            raise ValueError(f"{path} has only {len(self.data)} tokens; need more than block_size={block_size}")
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        chunk = np.stack([self.data[s: s + self.block_size + 1].astype(np.int64) for s in starts])
        t = torch.from_numpy(chunk)
        x, y = t[:, :-1], t[:, 1:]
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode train/val/test jsonl into uint16 token files")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/8192/merges.json")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/tokens/8192")
    args = ap.parse_args()
    tok = BPETokenizer.load(Path(args.tokenizer))
    for split in ("train", "val", "test"):
        n = encode_split(Path(args.data) / f"{split}.jsonl", tok, Path(args.out) / f"{split}.bin")
        print(f"{split}: {n:,} tokens")


if __name__ == "__main__":
    main()
