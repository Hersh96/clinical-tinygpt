"""Byte-level Byte Pair Encoding (BPE), written from scratch.

What a tokenizer does
---------------------
A model can't read letters; it reads integers. A tokenizer turns text into a
list of integer ids ("tokens") and back. Byte-level BPE, used by GPT-2 and most
modern LLMs, works like this:

1. Start from the 256 possible byte values. Any text, in any language, with
   any emoji, can be written as UTF-8 bytes, so nothing is ever "unknown".
2. Split text into chunks with a regex (words, numbers, punctuation, spaces),
   so merges never glue the end of one word onto the next.
3. Count every pair of neighbouring tokens across the training text. Merge the
   most frequent pair into a new token, e.g. ``"m" + "g" -> "mg"``. Repeat until
   the vocabulary reaches the size we asked for.
4. To encode new text, replay the merges in the order they were learned.

A bigger vocabulary means fewer tokens per word (shorter sequences, faster
models) but a bigger embedding table. Part 1 compares 4k, 8k and 16k.

The last id is reserved for ``<|endoftext|>``, which separates documents.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import regex

# GPT-2's pre-tokenization pattern: contractions, letters, numbers, other
# symbols, and whitespace (trailing whitespace is kept with the next word).
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pattern: str = GPT2_PATTERN):
        self.pattern = pattern
        self._re = regex.compile(pattern)
        self.merges: list[tuple[int, int]] = []
        self.special: dict[str, int] = {}
        self._rebuild()

    # ------------------------------------------------------------------ state
    def _rebuild(self) -> None:
        """Recompute the lookup tables from ``self.merges``."""
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        for text, idx in self.special.items():
            self.vocab[idx] = text.encode("utf-8")
        self._cache: dict[str, list[int]] = {}

    # --------------------------------------------------------------- training
    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        """Learn ``vocab_size - 257`` merges from ``texts`` (256 bytes + merges + EOT)."""
        n_merges = vocab_size - 256 - 1
        if n_merges < 0:
            raise ValueError("vocab_size must be at least 257")
        # Count each distinct chunk once, weighted by how often it appears.
        chunk_freq: Counter[str] = Counter()
        for text in texts:
            chunk_freq.update(self._re.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_freq]
        freqs = list(chunk_freq.values())

        # pair -> total count, and pair -> which words contain it.
        counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for w, (ids, f) in enumerate(zip(words, freqs)):
            for p in zip(ids, ids[1:]):
                counts[p] += f
                where[p].add(w)
        # Max-heap by count; ties go to the smallest pair, so training is deterministic.
        heap = [(-c, p) for p, c in counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if counts.get(pair, 0) != -neg or -neg <= 0:
                continue  # stale heap entry: the count changed since it was pushed
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for w in list(where[pair]):
                ids, f = words[w], freqs[w]
                for p in zip(ids, ids[1:]):       # remove this word's old pairs
                    counts[p] -= f
                    changed.add(p)
                ids = _merge(ids, pair, new_id)
                words[w] = ids
                for p in zip(ids, ids[1:]):       # add its new pairs
                    counts[p] += f
                    where[p].add(w)
                    changed.add(p)
            del counts[pair]
            where.pop(pair, None)
            for p in changed:
                if counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-counts[p], p))
        self.merges = merges
        self.special = {EOT: vocab_size - 1}
        self._rebuild()

    # --------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present in this chunk.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Text -> token ids. ``<|endoftext|>`` is one token only if ``allow_special``."""
        if not allow_special or not self.special:
            return self._encode_ordinary(text)
        out: list[int] = []
        parts = regex.split("(" + "|".join(regex.escape(s) for s in self.special) + ")", text)
        for part in parts:
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text (invalid UTF-8 from a partial sequence becomes U+FFFD)."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, text: str) -> list[str]:
        """The text of each token, for showing how a sentence is split."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"vocab_size": len(self.vocab), "pattern": self.pattern,
                "merges": [list(p) for p in self.merges], "special": self.special}
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(data["pattern"])
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.special = {k: int(v) for k, v in data["special"].items()}
        tok._rebuild()
        return tok
