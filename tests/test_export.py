import json

import pytest
import torch

from ctg.export.site_data import EXPECTED_FILES, export
from ctg.model.config import GPTConfig
from ctg.model.gpt import GPT
from ctg.tokenizer.bpe import BPETokenizer


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    import mlflow
    art = tmp_path / "artifacts"
    notes = [json.loads(l)["note"] for l in open("tests/fixtures/notes.jsonl")]
    for v in (300, 320, 340):
        t = BPETokenizer(); t.train(notes, v); t.save(art / "tokenizer" / str(v) / "merges.json")
    tok = BPETokenizer.load(art / "tokenizer" / "320" / "merges.json")
    (art / "tokenizer_compare.json").write_text(json.dumps({"ours_320": {"tokens_per_word": 1.3}, "gpt2": {"tokens_per_word": 1.6}}))
    (art / "scaling_fit.json").write_text(json.dumps({"points": [{"run": "gpt_5m", "n_params": 5e6, "val_loss": 3.1}],
                                                     "fit": {"a": 40, "alpha": 0.2, "c": 1.5, "r2": 0.99}}))
    (art / "bench_kv.json").write_text(json.dumps({"rows": [{"length": 512, "cache": True, "tokens_per_sec": 300},
                                                            {"length": 512, "cache": False, "tokens_per_sec": 60}]}))
    (art / "bpb.json").write_text(json.dumps({"gpt_85m": {"bpb": 1.1, "n_params": 85e6}, "gpt2_small": {"bpb": 1.3, "n_params": 124e6}}))
    (art / "samples.json").write_text(json.dumps({"gpt_40m": [{"prompt": "p", "output": "p q"}]}))
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri); mlflow.set_experiment("pretrain")
    for name in ("gpt_5m", "gpt_15m"):
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"sweep": "scaling_v1"})
            from mlflow.entities import Metric
            ms = [Metric("train_loss", 5 - s / 500, 0, s) for s in range(1, 1001)]
            ms += [Metric("val_loss", 5.1 - s / 500, 0, s) for s in range(50, 1001, 50)]
            client = mlflow.tracking.MlflowClient()
            for i in range(0, len(ms), 500):
                client.log_batch(run.info.run_id, metrics=ms[i:i + 500])
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=320, n_layer=2, n_head=2, n_embd=32, block_size=64)
    m = GPT(cfg)
    (tmp_path / "ckpt").mkdir()
    torch.save({"model": m.state_dict(), "model_config": cfg.to_dict()}, tmp_path / "ckpt" / "last.pt")
    return {"artifacts_dir": art, "mlflow_uri": uri, "ckpt": tmp_path / "ckpt" / "last.pt",
            "tokenizer": art / "tokenizer" / "320" / "merges.json", "out_dir": tmp_path / "site"}


def test_writes_every_file(workspace):
    paths = export(**workspace)
    names = {p.name for p in paths}
    assert set(EXPECTED_FILES) <= names
    assert {"merges_300.json", "merges_320.json", "merges_340.json"} <= names


def test_files_are_small_and_versioned(workspace):
    for p in export(**workspace):
        assert p.stat().st_size < 1_000_000
        data = json.loads(p.read_text())
        assert data["schema_version"] == 1 and "generated_at" in data


def test_loss_curves_are_downsampled(workspace):
    export(**workspace)
    curves = json.loads((workspace["out_dir"] / "loss_curves.json").read_text())["runs"]
    assert set(curves) == {"gpt_5m", "gpt_15m"}
    assert all(len(c["train"]) <= 300 for c in curves.values())
    assert curves["gpt_5m"]["train"][-1][0] == 1000  # keeps the final step


def test_attention_rows_sum_to_one(workspace):
    export(**workspace)
    a = json.loads((workspace["out_dir"] / "attention_example.json").read_text())
    T = len(a["tokens"])
    assert len(a["attention"]) == 2 and len(a["attention"][0]) == 2
    for layer in a["attention"]:
        for head in layer:
            assert len(head) == T
            assert all(abs(sum(row) - 1) < 0.01 for row in head)
    assert len(a["next_token"][-1]) == 10


def test_summary_headlines(workspace):
    export(**workspace)
    s = json.loads((workspace["out_dir"] / "summary.json").read_text())
    assert s["kv_speedup_512"] == pytest.approx(5.0)
    assert s["best_bpb"]["model"] == "gpt_85m" and s["gpt2_bpb"] == pytest.approx(1.3)


def test_missing_artifact_is_skipped_not_fatal(workspace):
    (workspace["artifacts_dir"] / "bench_kv.json").unlink()
    names = {p.name for p in export(**workspace)}
    assert "kv_bench.json" not in names and "summary.json" in names


def test_preview_flag_marks_summary(workspace):
    export(**workspace, preview=True)
    assert json.loads((workspace["out_dir"] / "summary.json").read_text())["preview"] is True
    export(**workspace)
    assert "preview" not in json.loads((workspace["out_dir"] / "summary.json").read_text())


def test_loss_curve_keeps_latest_value_per_step(workspace):
    """After a hard kill and resume, steps since the last checkpoint are logged twice."""
    import mlflow
    from mlflow.entities import Metric
    mlflow.set_tracking_uri(workspace["mlflow_uri"])
    run = mlflow.search_runs(experiment_names=["pretrain"], filter_string="tags.mlflow.runName = 'gpt_5m'",
                             output_format="list")[0]
    client = mlflow.tracking.MlflowClient()
    client.log_batch(run.info.run_id, metrics=[Metric("val_loss", 99.0, 10**13, s) for s in (950, 1000)])
    export(**workspace)
    val = json.loads((workspace["out_dir"] / "loss_curves.json").read_text())["runs"]["gpt_5m"]["val"]
    steps = [s for s, _ in val]  # 20 points: no downsampling, so duplicates would show
    assert len(steps) == len(set(steps)) == 20
    assert dict(map(tuple, val))[1000] == 99.0
