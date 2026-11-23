"""Scaling: how does loss improve as the model gets bigger?

We train four models (≈5M, 15M, 40M, 85M non-embedding parameters) on the same
data and token budget and fit the classic power law from the scaling-law papers:

    L(N) = a · N^(-alpha) + c

``alpha`` says how fast loss falls with size; ``c`` is the floor that more
parameters alone can't get below (for this data and budget). The fitted curve
lets the website's slider predict the loss of a model size we never trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

SIZES = ["5m", "15m", "40m", "85m"]
N0 = 1e6  # fit in units of millions of parameters, for numerical stability


def fit_power_law(n_params: list[float], losses: list[float]) -> dict:
    if len(n_params) < 3:
        raise ValueError("need at least 3 points to fit a, alpha and c")
    x, y = np.asarray(n_params, float) / N0, np.asarray(losses, float)

    def f(x, a, alpha, c):
        return a * x ** (-alpha) + c

    p0 = (y.max() - y.min() + 1e-3, 0.3, max(0.0, y.min() * 0.8))
    (a, alpha, c), _ = curve_fit(f, x, y, p0=p0, bounds=([0, 1e-4, 0], [np.inf, 1.0, np.inf]), maxfev=20000)
    resid = y - f(x, a, alpha, c)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return {"a": float(a * N0 ** alpha), "alpha": float(alpha), "c": float(c), "r2": r2}


def predict(fit: dict, n: float) -> float:
    return fit["a"] * n ** (-fit["alpha"]) + fit["c"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the four sizes (resumable), then fit the scaling law")
    ap.add_argument("--skip-train", action="store_true", help="only fit, from finished MLflow runs")
    ap.add_argument("--out", default="artifacts/scaling_fit.json")
    args = ap.parse_args()
    from ctg.tracking import setup
    from ctg.train.loop import load_config, train

    if not args.skip_train:
        for size in SIZES:
            print(f"=== training {size} ===")
            train(load_config(f"configs/model_{size}.yaml"))
    mlflow = setup(experiment="pretrain")
    runs = mlflow.search_runs(filter_string="tags.sweep = 'scaling_v1' and metrics.final_val_loss > 0",
                              order_by=["start_time DESC"])
    latest = runs.drop_duplicates("tags.mlflow.runName")
    pts = sorted((float(r["params.n_params"]), float(r["metrics.final_val_loss"]), r["tags.mlflow.runName"])
                 for _, r in latest.iterrows())
    fit = fit_power_law([p[0] for p in pts], [p[1] for p in pts])
    result = {"points": [{"run": r, "n_params": n, "val_loss": l} for n, l, r in pts], "fit": fit}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    with mlflow.start_run(run_name="scaling_fit"):
        mlflow.log_metrics({"alpha": fit["alpha"], "c": fit["c"], "r2": fit["r2"]})
        mlflow.log_artifact(args.out)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
