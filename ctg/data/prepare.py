"""Download, clean and split the text corpus.

Why this step matters
---------------------
A language model only learns from the text we feed it, and every number we
report later (loss, bits per byte, summary quality) is only honest if the
test documents were never seen during training. So this module does three
things, once, for both parts of the project:

1. Loads each source listed in ``configs/data.yaml`` (Hugging Face datasets or
   local jsonl files) and refuses to continue if a source has no license.
2. Cleans the text lightly (line endings and runs of spaces) without touching
   clinical symbols such as "µg" or "°C".
3. Assigns every document to train / val / test with a *hash of its id*, so
   the split is identical on every machine and every re-run (90 / 5 / 5).

Output: ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` where every line is
``{"id", "source", "text", "summary"}`` (``summary`` is ``None`` when a source
has no reference summary), plus ``LICENSES.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator, Literal

import yaml

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")


def split_for(doc_id: str, seed: int) -> Split:
    """Deterministically map a document id to a split (90/5/5)."""
    bucket = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest(), 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(s: str) -> str:
    """Normalise line endings and whitespace; keep every other character."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _SPACES.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _BLANKS.sub("\n\n", s)
    return s.strip()


def _iter_source(src: dict) -> Iterator[tuple[str, str, str | None]]:
    """Yield (raw_id, text, summary) for one configured source."""
    text_f, sum_f, id_f = src["text_field"], src.get("summary_field"), src.get("id_field")
    if src["kind"] == "jsonl":
        rows = (json.loads(line) for line in open(src["path"], encoding="utf-8"))
    elif src["kind"] == "hf":
        from datasets import load_dataset  # imported lazily: tests never hit the network

        rows = load_dataset(src["dataset"], src.get("config"), split=src.get("split", "train"), streaming=True)
    else:
        raise ValueError(f"unknown source kind {src['kind']!r}")
    max_chars = src.get("max_chars")
    used = 0
    for i, row in enumerate(rows):
        text = row[text_f]
        if not text:
            continue
        yield (str(row[id_f]) if id_f else str(i)), text, (row[sum_f] if sum_f else None)
        used += len(text)
        if max_chars and used >= max_chars:
            break


def prepare_corpus(cfg: dict, out_dir: Path) -> dict[str, Path]:
    """Build the split jsonl files described by ``cfg`` inside ``out_dir``."""
    out_dir = Path(out_dir)
    for src in cfg["sources"]:
        if not src.get("license"):
            raise ValueError(f"source {src['name']!r} has no license recorded; check it before use")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: out_dir / f"{s}.jsonl" for s in SPLITS}
    files = {s: open(p, "w", encoding="utf-8") for s, p in paths.items()}
    try:
        for src in cfg["sources"]:
            for raw_id, text, summary in _iter_source(src):
                doc_id = f"{src['name']}:{raw_id}"
                rec = {"id": doc_id, "source": src["name"], "text": clean_text(text),
                       "summary": clean_text(summary) if summary else None}
                files[split_for(doc_id, cfg["seed"])].write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for f in files.values():
            f.close()
    lic = ["# Data sources and licenses", ""] + [
        f"- **{s['name']}** ({s.get('dataset') or s.get('path')}): {s['license']}" for s in cfg["sources"]]
    (out_dir / "LICENSES.md").write_text("\n".join(lic) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for split, path in prepare_corpus(cfg, Path(args.out)).items():
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"{split}: {n} documents -> {path}")


if __name__ == "__main__":
    main()
