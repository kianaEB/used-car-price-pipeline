"""Fit a price regressor plus honest baselines, and save the trained models.

Regression only -- never a classifier (that was the prototype's core bug). Every model is compared
against a mean baseline so the reported metric is honest. random_state/params are threaded from the
config seed, so a run is reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import logging

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.config import load_settings
from src.features.preprocess import SplitData

log = logging.getLogger("model.train")

BASELINE_NAME = (
    "mean_baseline"  # the name every candidate is compared against for honesty
)


def build_models(cfg: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Return the model zoo to compare: mean baseline, linear, decision tree, random forest."""
    return {
        BASELINE_NAME: DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            random_state=seed, **cfg.get("decision_tree", {})
        ),
        "random_forest": RandomForestRegressor(
            random_state=seed, **cfg.get("random_forest", {})
        ),
    }


def train(
    split: SplitData,
    cfg: dict[str, Any],
    seed: int = 42,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit every model on the training split and save a joblib bundle; return {name: estimator}."""
    models = build_models(cfg, seed)
    for model in models.values():
        model.fit(split.X_train, split.y_train)

    path = (
        Path(artifact_path)
        if artifact_path is not None
        else load_settings().path("model")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "encoders": split.encoders}, path)
    log.info("trained %d models; saved bundle -> %s", len(models), path)
    return models
