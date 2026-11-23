"""One place that decides where MLflow stores runs.

Default: a local SQLite file ``mlflow.db`` (browse it with
``mlflow ui --backend-store-uri sqlite:///mlflow.db``). Override with the
``MLFLOW_TRACKING_URI`` environment variable or a config's ``mlflow_uri``.
"""
from __future__ import annotations

import os

DEFAULT_URI = "sqlite:///mlflow.db"


def setup(uri: str | None = None, experiment: str | None = None):
    import mlflow

    mlflow.set_tracking_uri(uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_URI)
    if experiment:
        mlflow.set_experiment(experiment)
    return mlflow
