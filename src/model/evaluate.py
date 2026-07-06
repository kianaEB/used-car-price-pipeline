"""Score fitted models on the held-out test split and save metrics.json.

Metrics: MAE, RMSE, R2, MAPE. The winner must beat the mean baseline -- or the log/README says so
plainly, rather than massaging numbers. `score()` is unit-tested; `evaluate()` wires it to the
models, picks the winner, and returns every model's metrics for the pipeline to record.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.features.preprocess import SplitData
from src.model.train import BASELINE_NAME

log = logging.getLogger("model.evaluate")


def score(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MAE, RMSE, R2, MAPE for one model's predictions (MAPE is zero-guarded by sklearn)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def winner(metrics: dict[str, dict[str, float]]) -> str:
    """Name of the best model: the lowest MAE on the test split."""
    return min(metrics, key=lambda name: metrics[name]["mae"])


def evaluate(
    models: dict[str, Any], split: SplitData, out_path: str | Path
) -> dict[str, dict[str, float]]:
    """Score every model on the test split, save metrics.json, and return {name: metrics}."""
    per_model = {
        name: score(split.y_test, model.predict(split.X_test))
        for name, model in models.items()
    }
    best = winner(per_model)
    baseline_mae = per_model.get(BASELINE_NAME, {}).get("mae", float("inf"))
    beats_baseline = best != BASELINE_NAME and per_model[best]["mae"] < baseline_mae

    report = {
        "models": per_model,
        "winner": best,
        "beats_baseline": beats_baseline,
        "baseline_mae": baseline_mae,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    _log_summary(per_model, best, baseline_mae, beats_baseline)
    return per_model


def _log_summary(
    per_model: dict[str, dict[str, float]],
    best: str,
    baseline_mae: float,
    beats_baseline: bool,
) -> None:
    """Log a one-line-per-model metric table and an honest winner-vs-baseline verdict."""
    for name, m in per_model.items():
        log.info(
            "%-18s MAE=%12.2f RMSE=%12.2f R2=%6.3f MAPE=%6.3f",
            name,
            m["mae"],
            m["rmse"],
            m["r2"],
            m["mape"],
        )
    if beats_baseline:
        improvement = (baseline_mae - per_model[best]["mae"]) / baseline_mae * 100
        log.info(
            "winner: %s (MAE %.2f, %.1f%% better than the mean baseline)",
            best,
            per_model[best]["mae"],
            improvement,
        )
    else:
        log.warning("no model beat the mean baseline; winner is %s", best)
