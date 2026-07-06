"""Tests for metric scoring, deterministic training, and that the model beats the mean baseline.

Metrics are checked against hand/sklearn references; the winner (lowest test MAE) must beat the
mean baseline on synthetic data; and training is reproducible for a fixed seed. A fast model config
keeps the suite quick; a separate test proves the real config params/seed are threaded through.
Uses stdlib tempfile for artifacts (the machine's pytest tmp_path base is locked).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from src.config import load_settings
from src.features.preprocess import SplitData, clean, split_and_encode
from src.ingest.dataset import generate_synthetic
from src.model.evaluate import evaluate, score, winner
from src.model.train import BASELINE_NAME, build_models, train

FAST_CFG = {"decision_tree": {"max_depth": 8}, "random_forest": {"n_estimators": 25}}
MODEL_NAMES = {BASELINE_NAME, "linear_regression", "decision_tree", "random_forest"}


def _split(n: int = 1500, seed: int = 42) -> SplitData:
    """A leakage-safe train/test split from clean synthetic data."""
    return split_and_encode(
        clean(generate_synthetic(n=n, seed=1, bad_fraction=0.0)), seed=seed
    )


# ------------------------------------------------------------------------------------- score


def test_score_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = score(y, y)
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_score_has_all_metrics():
    m = score([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert set(m) == {"mae", "rmse", "r2", "mape"}


def test_score_matches_reference():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 330.0, 380.0])
    errors = np.abs(y_true - y_pred)  # [10, 10, 30, 20]
    m = score(y_true, y_pred)
    assert m["mae"] == pytest.approx(errors.mean())  # 17.5
    assert m["rmse"] == pytest.approx(np.sqrt((errors**2).mean()))  # sqrt(375)
    assert m["r2"] == pytest.approx(r2_score(y_true, y_pred))
    assert m["mape"] == pytest.approx(mean_absolute_percentage_error(y_true, y_pred))


def test_mape_guards_against_zero_true_value():
    m = score([0.0, 100.0], [10.0, 90.0])  # a zero true value must not divide-by-zero
    assert np.isfinite(m["mape"])


# --------------------------------------------------------------------------------- build_models


def test_build_models_threads_config_params_and_seed():
    models = build_models(load_settings()["model"], seed=42)
    assert set(models) == MODEL_NAMES
    assert (
        models["random_forest"].n_estimators == 200
    )  # from config.model.random_forest
    assert models["decision_tree"].max_depth == 12  # from config.model.decision_tree
    assert models["decision_tree"].random_state == 42  # seed threaded through


# --------------------------------------------------------------------------------------- train


def test_train_fits_and_saves_a_loadable_bundle():
    split = _split()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        models = train(split, FAST_CFG, seed=42, artifact_path=path)
        assert set(models) == MODEL_NAMES
        assert path.exists()
        bundle = joblib.load(path)
    assert set(bundle["models"]) == MODEL_NAMES
    assert "preprocessor" in bundle["encoders"]
    models["random_forest"].predict(split.X_test)  # fitted -> predicts without error


def test_training_is_deterministic_for_a_fixed_seed():
    split = _split()
    with tempfile.TemporaryDirectory() as directory:
        a = train(split, FAST_CFG, seed=42, artifact_path=Path(directory) / "a.joblib")
        b = train(split, FAST_CFG, seed=42, artifact_path=Path(directory) / "b.joblib")
    np.testing.assert_array_equal(
        a["random_forest"].predict(split.X_test),
        b["random_forest"].predict(split.X_test),
    )


# ------------------------------------------------------------------------------------ evaluate


def test_evaluate_writes_all_model_metrics_and_picks_winner():
    split = _split()
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.joblib"
        metrics_path = Path(directory) / "metrics.json"
        models = train(split, FAST_CFG, seed=42, artifact_path=model_path)
        metrics = evaluate(models, split, metrics_path)
        report = json.loads(metrics_path.read_text())
    assert set(metrics) == MODEL_NAMES
    assert all(set(m) == {"mae", "rmse", "r2", "mape"} for m in metrics.values())
    assert set(report["models"]) == MODEL_NAMES
    assert report["winner"] == winner(metrics)
    assert report["beats_baseline"] is True


def test_best_model_beats_mean_baseline():
    split = _split()
    with tempfile.TemporaryDirectory() as directory:
        models = train(
            split, FAST_CFG, seed=42, artifact_path=Path(directory) / "m.joblib"
        )
        metrics = evaluate(models, split, Path(directory) / "metrics.json")
    best = winner(metrics)
    assert best != BASELINE_NAME
    assert metrics[best]["mae"] < metrics[BASELINE_NAME]["mae"]
    assert metrics[best]["r2"] > 0.0


def test_evaluate_is_deterministic_for_a_fixed_seed():
    split = _split()
    with tempfile.TemporaryDirectory() as directory:
        m1 = train(split, FAST_CFG, seed=42, artifact_path=Path(directory) / "a.joblib")
        m2 = train(split, FAST_CFG, seed=42, artifact_path=Path(directory) / "b.joblib")
        e1 = evaluate(m1, split, Path(directory) / "e1.json")
        e2 = evaluate(m2, split, Path(directory) / "e2.json")
    assert e1 == e2
