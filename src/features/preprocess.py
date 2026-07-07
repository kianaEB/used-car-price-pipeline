"""Clean, encode, and split the validated data -- with no leakage.

Encoders/scalers are fit on TRAIN ONLY, then applied to test. This is a correctness fix over the
prototype, which encoded across the whole dataset (and even mixed in the user's input row).
Imputation lives inside the fitted transformer (fit on train), so filling missing values never
leaks test statistics; unseen categories are ignored at transform time, not learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Settings, load_settings


@dataclass
class SplitData:
    """Train/test matrices plus the encoders fit on the training split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    encoders: dict[str, Any]


def clean(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Select the config feature+target columns from already-validated data and coerce dtypes.

    Does NOT impute -- imputation happens inside the train-fitted transformer to avoid leakage.
    Rows with a missing target are dropped (they cannot train or score a regressor).
    """
    settings = settings if settings is not None else load_settings()
    features = settings["features"]
    target = features["target"]
    numeric = [c for c in features["numeric"] if c in df.columns]
    categorical = [c for c in features["categorical"] if c in df.columns]

    keep = ([target] if target in df.columns else []) + numeric + categorical
    out = df[keep].copy()
    for col in numeric + ([target] if target in out.columns else []):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical:
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    if target in out.columns:
        out = out[out[target].notna()].reset_index(drop=True)
    return out


def _build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    max_categories: int | None = None,
) -> ColumnTransformer:
    """A ColumnTransformer: median-impute + scale numerics, constant-impute + one-hot categoricals.

    `max_categories` caps one-hot width by bucketing rare categories as 'infrequent' -- essential for
    real high-cardinality fields (e.g. free-text model); None keeps every category (synthetic).
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=max_categories,
                ),
            ),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")


def split_and_encode(
    df: pd.DataFrame,
    target: str = "price",
    test_size: float = 0.2,
    seed: int = 42,
    max_categories: int | None = None,
) -> SplitData:
    """Train/test split, then fit encoders/scalers on TRAIN ONLY and transform both (no leakage).

    Numeric columns are inferred by dtype, the rest are treated as categoricals. The fitted
    ColumnTransformer is returned in `encoders` so train/evaluate/predict reuse it unchanged;
    unseen test/inference categories are ignored (all-zero), never learned. Deterministic under seed.
    """
    y = df[target]
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols, max_categories)
    X_train_enc = preprocessor.fit_transform(X_train)  # FIT ON TRAIN ONLY
    X_test_enc = preprocessor.transform(
        X_test
    )  # transform with the train-fitted encoders
    feature_names = list(preprocessor.get_feature_names_out())

    return SplitData(
        X_train=pd.DataFrame(X_train_enc, columns=feature_names).reset_index(drop=True),
        X_test=pd.DataFrame(X_test_enc, columns=feature_names).reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        encoders={
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
    )
