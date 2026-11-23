"""How well does each tokenizer fit clinical text?

Measures, on held-out validation notes:
- tokens per word (lower = the tokenizer knows the domain's words better)
- characters per token (higher = more text per token = shorter sequences)
- how ten fixed clinical terms get split, e.g. GPT-2 may cut
  "hydrochlorothiazide" into many pieces where our clinical BPE uses fewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

EXAMPLE_TERMS = ["metformin", "hydrochlorothiazide", "acetaminophen", "bid", "prn",
                 "dyspnea", "tachycardia", "hba1c", "mg/dL", "status post"]


def compare(tokenizers: dict[str, Callable[[str], list[int]]], texts: list[str],
            pieces: dict[str, Callable[[str], list[str]]] | None = None) -> dict[str, dict]:
    words = sum(len(t.split()) for t in texts)
    chars = sum(len(t) for t in texts)
    out = {}
    for name, enc in tokenizers.items():
        n_tok = sum(len(enc(t)) for t in texts)
        to_pieces = (pieces or {}).get(name)
        out[name] = {
            "tokens_per_word": n_tok / words if words else 0.0,
            "chars_per_token": chars / n_tok if n_tok else 0.0,
            "n_tokens": n_tok,
            "examples": [{"term": term, "n_tokens": len(enc(" " + term)),
                          "pieces": to_pieces(term) if to_pieces else None} for term in EXAMPLE_TERMS],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--tokdir", default="artifacts/tokenizer")
    ap.add_argument("--n_docs", type=int, default=2000)
    ap.add_argument("--out", default="artifacts/tokenizer_compare.json")
    args = ap.parse_args()
    import mlflow
    import tiktoken

    from ctg.tokenizer.bpe import BPETokenizer

    texts = [json.loads(l)["text"] for _, l in zip(range(args.n_docs), open(args.val, encoding="utf-8"))]
    encs, pcs = {}, {}
    for d in sorted(Path(args.tokdir).iterdir(), key=lambda p: int(p.name)):
        tok = BPETokenizer.load(d / "merges.json")
        encs[f"ours_{d.name}"] = tok.encode
        pcs[f"ours_{d.name}"] = tok.token_strings
    g = tiktoken.get_encoding("gpt2")
    encs["gpt2"] = g.encode_ordinary
    pcs["gpt2"] = lambda s: [g.decode([i]) for i in g.encode_ordinary(s)]
    res = compare(encs, texts, pcs)
    Path(args.out).write_text(json.dumps(res, indent=1))
    mlflow.set_experiment("tokenizer")
    with mlflow.start_run(run_name="tokenizer_compare"):
        for name, r in res.items():
            mlflow.log_metric(f"{name}_tokens_per_word", r["tokens_per_word"])
        mlflow.log_artifact(args.out)
    for name, r in res.items():
        print(f"{name:12s} tokens/word {r['tokens_per_word']:.3f}  chars/token {r['chars_per_token']:.2f}")


if __name__ == "__main__":
    main()
