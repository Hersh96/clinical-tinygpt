"""Turn experiment results into small JSON files for the website guide.

The website never talks to MLflow or runs a model. Instead this script reads
the finished results (MLflow metric history and the JSON artifacts each
experiment wrote) and writes compact, versioned JSON files that the static
page loads. Re-running it after new training runs refreshes every chart.

A missing input (an experiment that hasn't run yet) is skipped with a
warning; the page shows "data not available yet" for that chapter.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

EXPECTED_FILES = ["summary.json", "tokenizer_compare.json", "loss_curves.json", "scaling.json",
                  "kv_bench.json", "bpb.json", "samples.json", "attention_example.json"]
EXAMPLE_SENTENCE = "The patient denies chest pain and was started on metformin."
MAX_POINTS = 300


def _write(out_dir: Path, name: str, payload: dict) -> Path:
    payload = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    p = out_dir / name
    p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return p


def _downsample(points: list, n: int = MAX_POINTS) -> list:
    if len(points) <= n:
        return points
    step = (len(points) - 1) / (n - 1)
    return [points[round(i * step)] for i in range(n)]


def _read(path: Path) -> dict | None:
    if not path.exists():
        print(f"skipping: {path} not found (run that experiment first)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def loss_curves(mlflow_uri: str | None) -> dict:
    from ctg.tracking import setup

    mlflow = setup(mlflow_uri, experiment="pretrain")
    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(filter_string="tags.sweep = 'scaling_v1'", order_by=["start_time DESC"], output_format="list")
    out = {}
    for run in runs:
        name = run.info.run_name
        if name in out:
            continue  # keep the latest run per size
        hist = {k: sorted((m.step, round(m.value, 4)) for m in client.get_metric_history(run.info.run_id, k))
                for k in ("train_loss", "val_loss")}
        out[name] = {"train": [list(p) for p in _downsample(hist["train_loss"])],
                     "val": [list(p) for p in _downsample(hist["val_loss"])],
                     "n_params": int(float(run.data.params.get("n_params", 0)))}
    return out


@torch.no_grad()
def attention_example(ckpt: Path, tokenizer: Path) -> dict:
    from ctg.experiments.common import load_checkpoint
    from ctg.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.load(tokenizer)
    model = load_checkpoint(str(ckpt), "cpu")
    ids = tok.encode(EXAMPLE_SENTENCE)
    logits, _, attns = model(torch.tensor([ids]), return_attn=True)
    probs = logits[0].float().softmax(-1)

    def top(row, k):
        v, i = row.topk(k)
        return [{"token": tok.vocab[j].decode("utf-8", "replace"), "p": round(p, 4)} for p, j in zip(v.tolist(), i.tolist())]

    emb = model.tok_emb.weight[ids, :16]
    return {
        "sentence": EXAMPLE_SENTENCE,
        "tokens": [tok.vocab[i].decode("utf-8", "replace") for i in ids],
        "embedding_preview": [[round(x, 3) for x in r] for r in (emb / emb.abs().max()).tolist()],
        "attention": [[[[round(x, 3) for x in row] for row in head] for head in layer[0].tolist()] for layer in attns],
        "next_token": [top(probs[t], 10 if t == len(ids) - 1 else 5) for t in range(len(ids))],
        "model": {k: model.cfg.to_dict()[k] for k in ("n_layer", "n_head", "n_embd")},
    }


def summary(tc: dict | None, sc: dict | None, kv: dict | None, bpb: dict | None) -> dict:
    s: dict = {}
    if tc:
        ours = {k: v for k, v in tc.items() if k.startswith("ours_")}
        best = min(ours, key=lambda k: ours[k]["tokens_per_word"])
        s["tokens_per_word"] = {"ours": ours[best]["tokens_per_word"], "ours_name": best,
                                "gpt2": tc["gpt2"]["tokens_per_word"]}
    if sc:
        s["scaling_alpha"] = sc["fit"]["alpha"]
        s["n_models"] = len(sc["points"])
    if kv:
        r = {(row["length"], row["cache"]): row["tokens_per_sec"] for row in kv["rows"]}
        L = max(length for length, _ in r)
        s["kv_speedup_512"] = r.get((512, True), r[(L, True)]) / r.get((512, False), r[(L, False)])
    if bpb:
        ours = {k: v for k, v in bpb.items() if k.startswith("gpt_")}
        best = min(ours, key=lambda k: ours[k]["bpb"])
        s["best_bpb"] = {"model": best, "bpb": ours[best]["bpb"], "n_params": ours[best]["n_params"]}
        if "gpt2_small" in bpb:
            s["gpt2_bpb"] = bpb["gpt2_small"]["bpb"]
    return s


def export(out_dir: Path, artifacts_dir: Path = Path("artifacts"), mlflow_uri: str | None = None,
           ckpt: Path = Path("ckpt/gpt_40m/last.pt"), tokenizer: Path = Path("artifacts/tokenizer/8192/merges.json"),
           preview: bool = False) -> list[Path]:
    out_dir, artifacts_dir = Path(out_dir), Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for d in sorted((artifacts_dir / "tokenizer").glob("*/merges.json")):
        merges = json.loads(d.read_text())
        written.append(_write(out_dir, f"merges_{d.parent.name}.json", merges))
    tc = _read(artifacts_dir / "tokenizer_compare.json")
    sc = _read(artifacts_dir / "scaling_fit.json")
    kv = _read(artifacts_dir / "bench_kv.json")
    bpb = _read(artifacts_dir / "bpb.json")
    samples = _read(artifacts_dir / "samples.json")
    for name, data in (("tokenizer_compare.json", {"tokenizers": tc}), ("scaling.json", sc), ("kv_bench.json", kv),
                       ("bpb.json", {"models": bpb}), ("samples.json", {"models": samples})):
        if data and all(v is not None for v in data.values()):
            written.append(_write(out_dir, name, data))
    curves = loss_curves(mlflow_uri)
    if curves:
        written.append(_write(out_dir, "loss_curves.json", {"runs": curves}))
    if Path(ckpt).exists() and Path(tokenizer).exists():
        written.append(_write(out_dir, "attention_example.json", attention_example(Path(ckpt), Path(tokenizer))))
    else:
        print(f"skipping attention example: {ckpt} or {tokenizer} not found")
    s = summary(tc, sc, kv, bpb)
    if preview:
        s["preview"] = True  # the page shows a "preview data" banner
    written.append(_write(out_dir, "summary.json", s))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/public/projects/clinical-tinygpt/data")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ckpt", default="ckpt/gpt_40m/last.pt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/8192/merges.json")
    ap.add_argument("--preview", action="store_true", help="mark the data as a preview (short test run)")
    args = ap.parse_args()
    for p in export(Path(args.out), Path(args.artifacts), None, Path(args.ckpt), Path(args.tokenizer), args.preview):
        print(f"{p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
