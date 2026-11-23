"""KV cache benchmark: generation speed and memory, with and without the cache.

Without a cache, producing token n means re-running the model over all n
earlier tokens, so each new token gets slower as the text grows. With a cache,
each step only processes the one new token. We measure tokens per second and
peak memory for several output lengths.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ctg.model.generate import generate


def bench(model, lengths: list[int] = [64, 128, 256, 512, 1024], repeats: int = 3, device="cpu") -> list[dict]:
    rows = []
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    for length in lengths:
        for cache in (True, False):
            best = 0.0
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                generate(model, prompt, length, temperature=0, use_cache=cache)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                best = max(best, length / (time.perf_counter() - t0))
            peak = torch.cuda.max_memory_allocated() / 2**20 if str(device).startswith("cuda") else None
            rows.append({"length": length, "cache": cache, "tokens_per_sec": best, "peak_mem_mb": peak})
            print(rows[-1])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="40m")
    ap.add_argument("--out", default="artifacts/bench_kv.json")
    args = ap.parse_args()
    from ctg.experiments.common import load_checkpoint
    from ctg.tracking import setup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(f"ckpt/gpt_{args.size}/last.pt", device)
    rows = bench(model, device=device)
    Path(args.out).write_text(json.dumps({"model": f"gpt_{args.size}", "device": device, "rows": rows}, indent=1))
    mlflow = setup(experiment="eval")
    with mlflow.start_run(run_name="bench_kv"):
        mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()
