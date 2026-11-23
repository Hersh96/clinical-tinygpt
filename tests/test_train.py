import numpy as np
import pytest
import torch

from ctg.train.loop import train


@pytest.fixture
def bins(tmp_path):
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 64, size=50).astype(np.uint16)
    np.tile(pattern, 40).tofile(tmp_path / "train.bin")   # highly repetitive: easy to learn
    np.tile(pattern, 10).tofile(tmp_path / "val.bin")
    return tmp_path


def cfg(tmp_path, **kw):
    c = {
        "run_name": "t", "seed": 0, "device": "cpu",
        "model": {"vocab_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 16},
        "data": {"train_bin": str(tmp_path / "train.bin"), "val_bin": str(tmp_path / "val.bin")},
        "batch_size": 8, "grad_accum": 1, "max_lr": 3e-3, "min_lr_ratio": 0.1, "warmup": 10,
        "total_steps": 100, "eval_every": 50, "eval_batches": 2, "log_every": 10,
        "ckpt_dir": str(tmp_path / "ckpt"), "ckpt_every_min": 10,
        "mlflow_uri": f"sqlite:///{tmp_path}/mlflow.db", "mlflow_experiment": "test", "resume": False,
    }
    c.update(kw)
    return c


def test_loss_drops_on_repetitive_data(bins):
    out = train(cfg(bins, total_steps=200, eval_every=200))
    assert out["final_val_loss"] < 0.5 * np.log(64)
    assert out["tokens_seen"] == 200 * 8 * 16
    assert out["params"] > 0 and out["run_id"]


def test_resume_matches_uninterrupted(bins, tmp_path):
    full = train(cfg(bins, ckpt_dir=str(tmp_path / "a"), run_name="full"))
    train(cfg(bins, ckpt_dir=str(tmp_path / "b"), run_name="cut", stop_after=60))
    resumed = train(cfg(bins, ckpt_dir=str(tmp_path / "b"), run_name="cut", resume=True))
    assert resumed["tokens_seen"] == full["tokens_seen"]
    assert resumed["final_val_loss"] == pytest.approx(full["final_val_loss"], abs=1e-6)
    a = torch.load(tmp_path / "a" / "full" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "cut" / "last.pt", weights_only=False)["model"]
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_checkpoint_has_everything_needed_to_resume(bins, tmp_path):
    train(cfg(bins, stop_after=20))
    ck = torch.load(tmp_path / "ckpt" / "t" / "last.pt", weights_only=False)
    assert {"model", "optimizer", "step", "stream", "rng", "config", "run_id"} <= set(ck)
    assert ck["step"] == 20


def test_resume_without_checkpoint_starts_fresh(bins):
    out = train(cfg(bins, resume=True, total_steps=5, eval_every=5))
    assert out["tokens_seen"] == 5 * 8 * 16


def test_preset_name_is_accepted(bins):
    from ctg.train.loop import build_model_config
    assert build_model_config({"preset": "5m"}).n_embd == 256
