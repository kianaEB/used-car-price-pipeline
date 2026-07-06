"""Score fitted models on the held-out test split and save metrics.json.

Metrics: MAE, RMSE, R2, MAPE. The winner must beat the mean baseline -- or the README says so
plainly. `score()` is implemented (and unit-tested); `evaluate()` wires it to the models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def score(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MAE, RMSE, R2, MAPE for one model's predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def evaluate(models: dict[str, Any], split: Any, out_path: str | Path) -> dict[str, dict]:
    """Score every fitted model on the test split, save metrics.json, and return the dict.

    TODO: predict on split.X_test; call score() per model; assemble {name: metrics}; write JSON
    to out_path; log the MAE winner and its improvement over the mean baseline.
    """
    raise NotImplementedError
