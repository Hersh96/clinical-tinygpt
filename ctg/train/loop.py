"""The training loop.

One optimisation *step*:
1. Sample ``batch_size`` random windows of tokens (``grad_accum`` times, so the
   effective batch is bigger than what fits in 6 GB at once).
2. Forward pass -> loss (how surprised the model is by the real next tokens).
3. Backward pass -> gradients; clip them to norm 1.0 so one bad batch can't
   blow up the weights.
4. AdamW update with the scheduled learning rate.

Everything needed to continue is checkpointed (weights, optimizer moments,
step, data-sampler RNG, torch RNG, MLflow run id), every ``ckpt_every_min``
minutes and when the run stops (including Ctrl+C or a laptop shutdown signal),
so ``resume: true`` continues exactly where it left off.
"""
from __future__ import annotations

import argparse
import contextlib
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ctg.data.tokens import TokenStream
from ctg.model.config import PRESETS, GPTConfig
from ctg.model.gpt import GPT
from ctg.tracking import setup as setup_tracking
from ctg.train.schedule import lr_at


def build_model_config(model_cfg: dict) -> GPTConfig:
    if "preset" in model_cfg:
        base = PRESETS[model_cfg["preset"]].to_dict()
        base.update({k: v for k, v in model_cfg.items() if k != "preset"})
        return GPTConfig(**base)
    return GPTConfig(**model_cfg)


def _optimizer(model: GPT, lr: float) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model: GPT, val_bin: str, block_size: int, batch_size: int, n_batches: int, device, autocast) -> float:
    """Mean loss on a fixed set of validation windows (same windows every time)."""
    model.eval()
    stream = TokenStream(Path(val_bin), block_size, seed=1234)
    losses = []
    for _ in range(n_batches):
        x, y = stream.batch(batch_size, device)
        with autocast():
            losses.append(model(x, y)[1].item())
    model.train()
    return float(np.mean(losses))


def train(cfg: dict) -> dict:
    import mlflow

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = build_model_config(cfg["model"])
    torch.manual_seed(cfg["seed"])
    model = GPT(mcfg).to(device)
    opt = _optimizer(model, cfg["max_lr"])
    stream = TokenStream(Path(cfg["data"]["train_bin"]), mcfg.block_size, seed=cfg["seed"])
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext

    ckpt_path = Path(cfg["ckpt_dir"]) / cfg["run_name"] / "last.pt"
    step, run_id = 0, None
    if cfg.get("resume") and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        stream.load_state_dict(ck["stream"])
        torch.set_rng_state(ck["rng"]["torch"])
        step, run_id = ck["step"], ck["run_id"]
        print(f"resumed {cfg['run_name']} at step {step}")

    def save():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "stream": stream.state_dict(), "rng": {"torch": torch.get_rng_state()},
                    "config": cfg, "model_config": mcfg.to_dict(), "run_id": run_id}, tmp)
        tmp.replace(ckpt_path)  # atomic: a crash mid-save never corrupts last.pt

    stop = {"flag": False}

    def _on_signal(signum, frame):
        print("signal received: saving checkpoint after this step")
        stop["flag"] = True

    old_handlers = {}
    if threading_main():
        for s in (signal.SIGINT, signal.SIGTERM):
            old_handlers[s] = signal.signal(s, _on_signal)

    setup_tracking(cfg.get("mlflow_uri"), cfg.get("mlflow_experiment", "pretrain"))
    total, end = cfg["total_steps"], min(cfg["total_steps"], cfg.get("stop_after", cfg["total_steps"]))
    tokens_per_step = cfg["batch_size"] * cfg["grad_accum"] * mcfg.block_size
    try:
        with mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg["run_name"]) as run:
            if run_id is None:
                run_id = run.info.run_id
                mlflow.log_params({**{k: v for k, v in cfg.items() if not isinstance(v, dict)},
                                   **{f"model.{k}": v for k, v in mcfg.to_dict().items()},
                                   "n_params": model.num_params(), "tokens_per_step": tokens_per_step})
                mlflow.set_tags(cfg.get("tags", {}))
            last_ckpt, t_log, val_loss = time.time(), time.time(), float("nan")
            model.train()
            while step < end and not stop["flag"]:
                lr = lr_at(step, cfg["max_lr"], cfg["warmup"], total, cfg["min_lr_ratio"])
                for g in opt.param_groups:
                    g["lr"] = lr
                loss_sum = 0.0
                for _ in range(cfg["grad_accum"]):
                    x, y = stream.batch(cfg["batch_size"], device)
                    with autocast():
                        loss = model(x, y)[1] / cfg["grad_accum"]
                    loss.backward()
                    loss_sum += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg["log_every"] == 0:
                    dt = time.time() - t_log
                    t_log = time.time()
                    m = {"train_loss": loss_sum, "lr": lr, "tokens_per_sec": cfg["log_every"] * tokens_per_step / dt}
                    if device.startswith("cuda"):
                        m["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    mlflow.log_metrics(m, step=step)
                if step % cfg["eval_every"] == 0 or step == total:
                    val_loss = evaluate(model, cfg["data"]["val_bin"], mcfg.block_size, cfg["batch_size"],
                                        cfg["eval_batches"], device, autocast)
                    mlflow.log_metric("val_loss", val_loss, step=step)
                    print(f"step {step}/{total} train {loss_sum:.3f} val {val_loss:.3f} lr {lr:.2e}")
                if time.time() - last_ckpt > cfg["ckpt_every_min"] * 60:
                    save()
                    last_ckpt = time.time()
            save()
            if step == total:
                mlflow.log_metric("final_val_loss", val_loss)
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)
    return {"run_id": run_id, "final_val_loss": val_loss, "tokens_seen": step * tokens_per_step,
            "params": model.num_params(), "step": step}


def threading_main() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


def load_config(path: str) -> dict:
    """Load a YAML config; ``base:`` names another config whose keys this one overrides."""
    cfg = yaml.safe_load(open(path))
    if "base" in cfg:
        base = load_config(str(Path(path).parent / cfg.pop("base")))
        for k, v in cfg.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        cfg = base
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["resume"] = cfg.get("resume", False) or args.resume
    print(train(cfg))


if __name__ == "__main__":
    main()
