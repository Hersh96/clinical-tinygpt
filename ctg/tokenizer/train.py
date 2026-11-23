"""Train the BPE tokenizer at every vocab size in configs/tokenizer.yaml."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ctg.tokenizer.bpe import BPETokenizer


def sample_texts(jsonl: Path, sample_mb: float) -> list[str]:
    """Read training documents until about ``sample_mb`` megabytes of text."""
    out, used, cap = [], 0, int(sample_mb * 1e6)
    for line in open(jsonl, encoding="utf-8"):
        text = json.loads(line)["text"]
        out.append(text)
        used += len(text.encode("utf-8"))
        if used >= cap:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    texts = sample_texts(Path(cfg["train_jsonl"]), cfg["sample_mb"])
    import mlflow
    from ctg.tracking import setup
    setup(experiment="tokenizer")
    for v in cfg["vocab_sizes"]:
        with mlflow.start_run(run_name=f"bpe_{v}"):
            t0 = time.time()
            tok = BPETokenizer()
            tok.train(texts, v)
            path = Path(cfg["out_dir"]) / str(v) / "merges.json"
            tok.save(path)
            mlflow.log_params({"vocab_size": v, "sample_mb": cfg["sample_mb"], "n_docs": len(texts)})
            mlflow.log_metric("train_seconds", time.time() - t0)
            mlflow.log_artifact(str(path))
            print(f"vocab {v}: {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
