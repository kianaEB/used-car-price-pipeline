"""Clean, encode, and split the validated data -- with no leakage.

Encoders/scalers are fit on TRAIN ONLY, then applied to test. This is a correctness fix over the
prototype, which encoded across the whole dataset (and even mixed in the user's input row).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SplitData:
    """Train/test matrices plus the encoders fit on the training split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    encoders: dict[str, Any]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select model columns and coerce dtypes on already-validated data.

    TODO: choose feature columns (brand, model, year, mileage, + available categoricals);
    coerce numeric dtypes; handle any remaining missing values with a documented strategy.
    """
    raise NotImplementedError


def split_and_encode(
    df: pd.DataFrame, target: str = "price", test_size: float = 0.2, seed: int = 42
) -> SplitData:
    """Train/test split, then fit categorical encoders on TRAIN ONLY and transform both.

    TODO: use sklearn.model_selection.train_test_split(random_state=seed); fit encoders on
    X_train, transform X_train/X_test; return SplitData carrying the fitted encoders so
    evaluation/prediction reuse them. Must be deterministic under `seed`.
    """
    raise NotImplementedError
