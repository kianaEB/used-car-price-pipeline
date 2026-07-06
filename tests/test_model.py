"""Tests for metric scoring and (once implemented) that the model beats the mean baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.evaluate import score


def test_score_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = score(y, y)
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_score_has_all_metrics():
    m = score([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert set(m) == {"mae", "rmse", "r2", "mape"}


@pytest.mark.skip(
    reason="TODO: implement preprocess+train+evaluate, then assert winner beats baseline"
)
def test_best_model_beats_mean_baseline():
    """Build a split from synthetic data, train, evaluate, assert best MAE < mean_baseline MAE."""
    ...
