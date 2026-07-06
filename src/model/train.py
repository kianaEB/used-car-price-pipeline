"""Fit a price regressor plus honest baselines, and save the best model.

Regression only -- never a classifier (that was the prototype's core bug). Every model is compared
against a mean baseline so the reported metric is honest.
"""
from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def build_models(cfg: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Return the model zoo to compare: mean baseline, linear, decision tree, random forest."""
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=seed, **cfg.get("decision_tree", {})),
        "random_forest": RandomForestRegressor(random_state=seed, **cfg.get("random_forest", {})),
    }


def train(split: Any, cfg: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Fit every model on the training split; return {name: fitted_estimator}.

    TODO: build models via build_models(cfg, seed); fit each on split.X_train / split.y_train;
    optionally persist the winner to artifacts/model.joblib (with the fitted encoders).
    """
    raise NotImplementedError
