"""Tests for src.ingest.dataset: canonical mapping, batch slicing, and injected-defect ground truth.

Covers the three things stage 2 promises: load_dataset renames source columns and coerces dtypes
without dropping rows; generate_synthetic is deterministic and injects the SPEC 6.3 defects at
~bad_fraction; iter_batches yields chronological, min-size slices with drift trending across weeks.
Uses stdlib tempfile (the machine's pytest tmp_path base has a locked ACL).
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import load_settings
from src.ingest.dataset import (
    generate_synthetic,
    iter_batches,
    load,
    load_dataset,
)

YEAR_RANGE = (1950, 2027)


@contextmanager
def _temp_csv(text: str) -> Iterator[Path]:
    """Yield the path to a throwaway CSV holding `text` (stdlib tempfile; auto-cleaned)."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vehicles.csv"
        path.write_text(text)
        yield path


# --------------------------------------------------------------------------- load_dataset


def test_load_dataset_maps_and_coerces_without_dropping_rows() -> None:
    """Source columns are renamed, numerics coerced (bad -> NaN), and no row is silently dropped."""
    csv = (
        "manufacturer,odometer,price,model,year,title_status,vin,posting_date\n"
        "toyota,45000,12000,corolla,2018,clean,V1,2021-01-05\n"
        "ford,not-a-number,abc,focus,,clean,V2,2021-01-06\n"
    )
    with _temp_csv(csv) as path:
        df = load_dataset(path)

    assert "brand" in df.columns and "manufacturer" not in df.columns
    assert "mileage" in df.columns and "odometer" not in df.columns
    assert len(df) == 2  # the garbage row is kept, not dropped
    assert pd.api.types.is_numeric_dtype(df["price"])
    assert pd.isna(df.loc[1, "price"]) and pd.isna(df.loc[1, "mileage"])
    assert pd.isna(df.loc[1, "year"])
    assert pd.api.types.is_datetime64_any_dtype(df["posting_date"])


def test_load_dataset_honours_custom_column_map() -> None:
    """A caller-supplied column_map overrides the default manufacturer/odometer mapping."""
    csv = "make,miles,price,model,year\nbmw,20000,30000,x5,2020\n"
    with _temp_csv(csv) as path:
        df = load_dataset(path, column_map={"make": "brand", "miles": "mileage"})
    assert {"brand", "mileage"}.issubset(df.columns)
    assert df.loc[0, "brand"] == "bmw"


# ----------------------------------------------------------------------- generate_synthetic


def test_generate_synthetic_is_deterministic() -> None:
    """Same seed -> identical frame; different seed -> different values (determinism knob works)."""
    a = generate_synthetic(n=500, seed=7)
    b = generate_synthetic(n=500, seed=7)
    c = generate_synthetic(n=500, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["price"].equals(c["price"])


def test_generate_synthetic_has_canonical_columns() -> None:
    """The generated frame carries every canonical column plus the optional categoricals."""
    df = generate_synthetic(n=300, seed=1)
    for col in ("price", "brand", "model", "year", "mileage", "title_status", "vin"):
        assert col in df.columns
    assert {"condition", "state", "posting_date"}.issubset(df.columns)
    assert len(df) == 300


def test_generate_synthetic_injects_defects_at_expected_rate() -> None:
    """Defects land at ~bad_fraction and each SPEC 6.3 defect type is actually present in the frame."""
    n, bad_fraction = 4000, 0.06
    df = generate_synthetic(n=n, seed=42, bad_fraction=bad_fraction)

    assert df.attrs["n_defect_rows"] == round(bad_fraction * n)
    assert abs(df.attrs["n_defect_rows"] / n - bad_fraction) < 0.01

    defects = df.attrs["defects"]
    # Clean rows never produce these signals, so counts match the recorded ground truth exactly.
    assert (df["price"] <= 0).sum() == defects["nonpositive_price"] > 0
    out_of_range_year = (df["year"] < YEAR_RANGE[0]) | (df["year"] > YEAR_RANGE[1])
    assert out_of_range_year.sum() == defects["impossible_year"] > 0
    assert df["brand"].isna().sum() == defects["null_brand"] > 0
    assert (df["vin"] == "DUPLICATE-VIN").sum() == defects["duplicate_vin"] >= 2
    assert df["vin"].duplicated().any()


def test_generate_synthetic_zero_bad_fraction_is_clean() -> None:
    """bad_fraction=0 injects nothing: no defect rows and no non-positive prices."""
    df = generate_synthetic(n=500, seed=3, bad_fraction=0.0)
    assert df.attrs["n_defect_rows"] == 0
    assert (df["price"] <= 0).sum() == 0
    assert df["brand"].isna().sum() == 0


def test_generate_synthetic_rising_null_drift() -> None:
    """The configured rising_null_column grows nullier over the weeks (null-rate drift signal)."""
    drift = {"rising_null_column": "mileage"}
    df = generate_synthetic(n=6000, seed=42, bad_fraction=0.0, n_weeks=8, drift=drift)
    week = (df["posting_date"] - df["posting_date"].min()).dt.days // 7
    first = df.loc[week == week.min(), "mileage"].isna().mean()
    last = df.loc[week == week.max(), "mileage"].isna().mean()
    assert last > first


def test_generate_synthetic_new_brand_category_shift() -> None:
    """A new brand is absent before new_brand_week and present afterwards (category-shift signal)."""
    df = generate_synthetic(
        n=6000, seed=42, bad_fraction=0.0, n_weeks=5, drift={"new_brand_week": 2}
    )
    week = (df["posting_date"] - df["posting_date"].min()).dt.days // 7
    before = set(df.loc[week < 2, "brand"])
    after = set(df.loc[week >= 2, "brand"])
    assert "rivian" not in before
    assert "rivian" in after


def test_generate_synthetic_price_shock_is_a_discrete_step() -> None:
    """The injected price shock is a one-time permanent step: mean price jumps once, flat elsewhere."""
    drift = {  # no gradual inflation, so the ONLY jump is the shock
        "price_inflation_per_week": 0.0,
        "price_shock_week": 4,
        "price_shock_multiplier": 1.5,
    }
    df = generate_synthetic(n=8000, seed=42, bad_fraction=0.0, n_weeks=8, drift=drift)
    week = (df["posting_date"] - df["posting_date"].min()).dt.days // 7
    means = df.groupby(week)["price"].mean()
    ratios = (means / means.shift(1)).dropna()
    assert ratios.loc[4] > 1.35  # a clear step up at the shock week
    assert (
        ratios.drop(index=4) < 1.1
    ).all()  # flat before and after (no gradual creep)


def test_load_dispatches_to_synthetic_source() -> None:
    """load() with the default (synthetic) config returns a non-empty canonical frame, no I/O."""
    df = load(load_settings())
    assert len(df) > 0
    assert {"price", "brand", "model", "year", "mileage"}.issubset(df.columns)


# --------------------------------------------------------------------------- iter_batches


def _batching(min_rows: int = 100) -> dict[str, object]:
    """Batching config matching config.yaml's batching block (weekly slices)."""
    return {"date_column": "posting_date", "freq": "W", "min_batch_rows": min_rows}


def test_iter_batches_are_chronological_and_min_sized() -> None:
    """Batches come out in date order, each at least min_batch_rows, one ISO week per batch."""
    df = generate_synthetic(n=3000, seed=42, bad_fraction=0.0, n_weeks=4)
    batches = list(iter_batches(df, _batching(min_rows=100)))

    assert len(batches) >= 3  # enough runs for a real history the dashboard can plot
    labels = [label for label, _ in batches]
    assert labels == sorted(labels)
    for _, batch in batches:
        assert len(batch) >= 100
        weeks = batch["posting_date"].dt.to_period("W").nunique()
        assert weeks == 1


def test_iter_batches_skips_undersized_slices() -> None:
    """A stale-date defect forms a tiny out-of-range slice that the min-rows filter drops."""
    df = generate_synthetic(n=3000, seed=42, bad_fraction=0.06, n_weeks=4)
    batches = list(iter_batches(df, _batching(min_rows=100)))
    stale = df[
        "posting_date"
    ].min()  # the injected stale rows sit years before the series
    assert all((batch["posting_date"] > stale).all() for _, batch in batches)


def test_iter_batches_reflect_price_inflation_drift() -> None:
    """With price_inflation_per_week > 0, later batches carry a higher mean price than earlier ones."""
    drift = {"price_inflation_per_week": 0.05}
    df = generate_synthetic(n=4000, seed=42, bad_fraction=0.0, n_weeks=4, drift=drift)
    batches = list(iter_batches(df, _batching(min_rows=100)))
    first_mean = batches[0][1]["price"].mean()
    last_mean = batches[-1][1]["price"].mean()
    assert last_mean > first_mean
