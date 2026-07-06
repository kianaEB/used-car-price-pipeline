"""Tests for leakage-safe, deterministic preprocessing."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement preprocess.split_and_encode (SPEC 8)")
def test_split_is_deterministic():
    """Same seed -> identical train/test partition."""
    ...


@pytest.mark.skip(reason="TODO: implement preprocess.split_and_encode (SPEC 8)")
def test_encoders_fit_on_train_only():
    """Encoders are fit on the training split; test-set categories never leak into the fit."""
    ...
