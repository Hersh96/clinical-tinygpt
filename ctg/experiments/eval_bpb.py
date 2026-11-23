"""Bits per byte: a fair way to compare models with different tokenizers.

Per-token perplexity depends on how big the tokens are: a tokenizer with
longer tokens makes fewer, harder predictions. So instead we add up the
model's total surprise (negative log-likelihood, in nats) over a text and
divide by the number of UTF-8 *bytes* in that text:

    bits per byte = total_nll_nats / (ln 2 · n_bytes)

A model that knows nothing about bytes scores 8.0 (one of 256 values per
byte). Lower is better, and the number means the same thing for our BPE
model and for GPT-2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def bits_per_byte(nll_sum_nats: float, n_bytes: int) -> float:
    return nll_sum_nats / (math.log(2) * n_bytes)


@torch.no_grad()
def nll_sum(logits_fn: Callable[[torch.Tensor], torch.Tensor], ids: list[int], block_size: int,
            device="cpu") -> float:
    """Total NLL (nats) of ids[1:] given what precedes them, in windows of ``block_size``.

    Windows overlap by one token, so every token after the first is predicted exactly once.
    """
    total, start = 0.0, 0
    while start < len(ids) - 1:
        chunk = torch.tensor([ids[start:start + block_size + 1]], device=device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits = logits_fn(x)
        total += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        start += block_size
    return total


def eval_bpb(model_fn: Callable[[str], float], texts: list[str]) -> float:
    """``model_fn(text)`` returns the text's total NLL in nats."""
    nll = sum(model_fn(t) for t in texts)
    return bits_per_byte(nll, sum(len(t.encode("utf-8")) for t in texts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--n_docs", type=int, default=500)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/8192/merges.json")
    ap.add_argument("--sizes", default="5m,15m,40m,85m")
    ap.add_argument("--out", default="artifacts/bpb.json")
    ap.add_argument("--samples_out", default="artifacts/samples.json")
    args = ap.parse_args()
    from ctg.experiments.common import PROMPTS, load_checkpoint
    from ctg.model.generate import generate
    from ctg.tokenizer.bpe import EOT, BPETokenizer
    from ctg.tracking import setup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [json.loads(l)["text"] for _, l in zip(range(args.n_docs), open(args.test, encoding="utf-8"))]
    tok = BPETokenizer.load(Path(args.tokenizer))
    eot = tok.special[EOT]
    results, samples = {}, {}
    for size in args.sizes.split(","):
        model = load_checkpoint(f"ckpt/gpt_{size}/last.pt", device)
        fn = lambda t, m=model: nll_sum(lambda x: m(x)[0], [eot] + tok.encode(t), m.cfg.block_size, device)
        results[f"gpt_{size}"] = {"bpb": eval_bpb(fn, texts), "n_params": model.num_params()}
        samples[f"gpt_{size}"] = [
            {"prompt": p, "output": tok.decode(generate(model, torch.tensor([tok.encode(p)], device=device), 60,
                                                        temperature=0.8, top_k=50, seed=0)[0].tolist())}
            for p in PROMPTS]
        print(size, results[f"gpt_{size}"])
    import tiktoken
    from transformers import GPT2LMHeadModel

    g = tiktoken.get_encoding("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    fn = lambda t: nll_sum(lambda x: gpt2(x).logits, [g.eot_token] + g.encode_ordinary(t), 1024, device)
    results["gpt2_small"] = {"bpb": eval_bpb(fn, texts), "n_params": sum(p.numel() for p in gpt2.parameters())}
    print("gpt2", results["gpt2_small"])
    Path(args.out).write_text(json.dumps(results, indent=1))
    Path(args.samples_out).write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    mlflow = setup(experiment="eval")
    with mlflow.start_run(run_name="bpb"):
        mlflow.log_metrics({f"{k}_bpb": v["bpb"] for k, v in results.items()})
        mlflow.log_artifact(args.out)
        mlflow.log_artifact(args.samples_out)


if __name__ == "__main__":
    main()
