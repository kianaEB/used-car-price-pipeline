"""Tests for leakage-safe, deterministic preprocessing.

Proves the split is reproducible under a seed, that every encoder/scaler is fit on TRAIN ONLY
(a test-only category never enters the fit and an unseen category never crashes transform), and
that the target is cleanly separated from the features. Params come from config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_settings
from src.features.preprocess import clean, split_and_encode
from src.ingest.dataset import generate_synthetic


def _clean_frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """A cleaned feature+target frame from clean synthetic data."""
    return clean(generate_synthetic(n=n, seed=seed, bad_fraction=0.0))


# ------------------------------------------------------------------------------------- clean


def test_clean_selects_config_features_and_drops_null_target():
    df = generate_synthetic(n=50, seed=1, bad_fraction=0.0)
    df.loc[0, "price"] = None  # a row with no target must be dropped
    cleaned = clean(df)
    assert {"price", "brand", "model", "year", "mileage", "condition", "state"} == set(
        cleaned.columns
    )
    assert "vin" not in cleaned.columns and "posting_date" not in cleaned.columns
    assert cleaned["price"].notna().all()
    assert len(cleaned) == 49


# ------------------------------------------------------------------------------- determinism


def test_split_is_deterministic_for_a_fixed_seed():
    cleaned = _clean_frame()
    a = split_and_encode(cleaned, seed=7)
    b = split_and_encode(cleaned, seed=7)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    pd.testing.assert_series_equal(a.y_train, b.y_train)


def test_different_seed_gives_a_different_partition():
    cleaned = _clean_frame()
    a = split_and_encode(cleaned, seed=7)
    c = split_and_encode(cleaned, seed=8)
    assert not a.y_train.equals(c.y_train)


def test_uses_config_test_size_and_seed():
    settings = load_settings()
    cleaned = _clean_frame(n=500)
    split = split_and_encode(
        cleaned, test_size=settings["split"]["test_size"], seed=settings.seed
    )
    expected_test = round(len(cleaned) * settings["split"]["test_size"])
    assert abs(len(split.X_test) - expected_test) <= 1


# --------------------------------------------------------------------------- target separation


def test_target_is_separated_from_features():
    cleaned = _clean_frame(n=300, seed=2)
    split = split_and_encode(cleaned, target="price", seed=3)
    assert all("price" not in column for column in split.X_train.columns)
    assert split.y_train.name == "price"
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_test) == len(split.y_test)
    assert len(split.X_train) + len(split.X_test) == len(cleaned)


# --------------------------------------------------------------------------------- no leakage


def test_encoders_fit_on_train_only():
    """The one-hot encoder learns exactly the TRAIN categories -- test rows never enter the fit."""
    seed = 5
    cleaned = _clean_frame(seed=3)
    split = split_and_encode(cleaned, seed=seed)
    X_train, X_test, _, _ = train_test_split(
        cleaned.drop(columns=["price"]),
        cleaned["price"],
        test_size=0.2,
        random_state=seed,
    )
    ohe = split.encoders["preprocessor"].named_transformers_["cat"].named_steps["ohe"]
    for i, col in enumerate(split.encoders["categorical"]):
        learned = set(ohe.categories_[i]) - {"__missing__"}
        train_values = set(X_train[col].dropna())
        test_only = set(X_test[col].dropna()) - train_values
        assert learned == train_values  # fit used the train split only
        assert learned.isdisjoint(test_only)  # any test-only category was excluded


def test_scaler_fit_on_train_only():
    """The scaler's learned mean matches the TRAIN numeric means, not the full dataset's."""
    seed = 5
    cleaned = _clean_frame(seed=3)
    split = split_and_encode(cleaned, seed=seed)
    X_train, _, _, _ = train_test_split(
        cleaned.drop(columns=["price"]),
        cleaned["price"],
        test_size=0.2,
        random_state=seed,
    )
    scaler = (
        split.encoders["preprocessor"].named_transformers_["num"].named_steps["scale"]
    )
    train_means = [X_train[col].mean() for col in split.encoders["numeric"]]
    full_means = [cleaned[col].mean() for col in split.encoders["numeric"]]
    assert np.allclose(scaler.mean_, train_means, rtol=1e-9)
    assert not np.allclose(
        scaler.mean_, full_means, rtol=1e-9
    )  # not the whole-data mean


def test_unseen_category_is_ignored_not_crashing():
    """A category never seen in training transforms without error (all-zero), never crashing."""
    cleaned = _clean_frame(seed=3)
    split = split_and_encode(cleaned, seed=5)
    preprocessor = split.encoders["preprocessor"]

    unseen = cleaned.drop(columns=["price"]).iloc[[0]].copy()
    unseen["brand"] = "__brand_never_seen__"
    transformed = preprocessor.transform(unseen)  # must not raise

    assert transformed.shape[1] == len(split.encoders["feature_names"])
    assert np.isfinite(transformed).all()
    brand_cols = [
        i
        for i, name in enumerate(split.encoders["feature_names"])
        if name.startswith("cat__brand_")
    ]
    assert transformed[0, brand_cols].sum() == 0  # unknown brand -> all-zero one-hot
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    assert "__brand_never_seen__" not in ohe.categories_[0]
