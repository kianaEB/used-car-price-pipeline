"""Tests for the Layer-2 business rules: each passes on clean data and fails on its own defect.

Every rule is checked for the right severity and n_violations, category/consistency rules skip
gracefully when their columns are absent, and offender samples must be JSON-serialisable (the
report writes them to disk).
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from src.quality import checks

REQUIRED = ["price", "brand", "model", "year", "mileage"]
RANGES = {"price": [1, 1_000_000], "year": [1950, 2027], "mileage": [0, 1_000_000]}
KNOWN_TITLES = ["clean", "salvage", "rebuilt", "lien", "missing", "parts only"]
FRESH = ["new", "like new"]
MAX_FRESH_MILEAGE = 60_000


# --------------------------------------------------------------------- columns / min_rows


def test_columns_pass_on_clean(clean_df):
    assert checks.check_schema(clean_df, REQUIRED).passed


def test_columns_fail_on_missing_column(clean_df):
    r = checks.check_schema(clean_df.drop(columns=["price"]), REQUIRED)
    assert not r.passed
    assert r.severity == "ERROR"
    assert r.n_violations == 1


def test_min_rows_fails_when_too_few(clean_df):
    assert not checks.check_min_rows(clean_df, min_rows=100).passed


def test_min_rows_passes_when_enough(clean_df):
    assert checks.check_min_rows(clean_df, min_rows=1).passed


# ------------------------------------------------------------------------------- ranges


def test_ranges_flag_negative_price_and_bad_year(dirty_df):
    r = checks.check_ranges(dirty_df, RANGES)
    assert not r.passed
    assert r.severity == "ERROR"
    assert r.n_violations >= 2  # negative price + impossible year


def test_ranges_pass_on_clean(clean_df):
    assert checks.check_ranges(clean_df, RANGES).passed


# -------------------------------------------------------------------------------- nulls


def test_nulls_flag_missing_brand(dirty_df):
    r = checks.check_nulls(dirty_df, {"brand": 0.0})
    assert not r.passed
    assert r.severity == "ERROR"  # zero-tolerance column escalates to ERROR
    assert r.n_violations >= 1


def test_nulls_warn_when_tolerance_column_breached(dirty_df):
    r = checks.check_nulls(dirty_df, {"brand": 0.10})  # 25% null > 10% tolerance
    assert not r.passed
    assert r.severity == "WARN"


def test_nulls_pass_on_clean(clean_df):
    assert checks.check_nulls(
        clean_df, {"brand": 0.0, "price": 0.0, "year": 0.0}
    ).passed


def test_nulls_skip_absent_column(clean_df):
    assert checks.check_nulls(clean_df, {"nonexistent": 0.0}).passed


def test_ranges_skip_absent_column(clean_df):
    assert checks.check_ranges(clean_df, {"nonexistent": [0, 1]}).passed


# --------------------------------------------------------------------------- duplicates


def test_duplicates_flag_repeated_vin(dirty_df):
    r = checks.check_duplicates(dirty_df)
    assert not r.passed
    assert r.severity == "ERROR"
    assert r.n_violations >= 1


def test_duplicates_pass_on_clean(clean_df):
    assert checks.check_duplicates(clean_df).passed


def test_duplicates_flag_fully_duplicated_row():
    df = pd.DataFrame({"price": [100.0, 100.0], "brand": ["x", "x"]})
    r = checks.check_duplicates(df)
    assert not r.passed
    assert r.n_violations == 2
    assert "duplicated row" in r.detail


# --------------------------------------------------------------------------- categories


def test_categories_warn_on_unknown_title_status():
    df = pd.DataFrame({"title_status": ["clean", "flooded", "salvage"]})
    r = checks.check_categories(df, "title_status", KNOWN_TITLES)
    assert not r.passed
    assert r.severity == "WARN"
    assert r.n_violations == 1


def test_categories_pass_when_all_known():
    df = pd.DataFrame({"title_status": ["clean", "salvage", "rebuilt"]})
    assert checks.check_categories(df, "title_status", KNOWN_TITLES).passed


def test_categories_skip_when_column_absent(clean_df):
    r = checks.check_categories(
        clean_df.drop(columns=["title_status"]), "title_status", KNOWN_TITLES
    )
    assert r.passed
    assert "skipped" in r.detail


# -------------------------------------------------------------------------- consistency


def test_consistency_warn_on_fresh_high_mileage():
    df = pd.DataFrame(
        {"condition": ["like new", "good"], "mileage": [220_000.0, 30_000.0]}
    )
    r = checks.check_consistency(df, FRESH, MAX_FRESH_MILEAGE)
    assert not r.passed
    assert r.severity == "WARN"
    assert r.n_violations == 1


def test_consistency_pass_when_consistent():
    df = pd.DataFrame(
        {"condition": ["like new", "good"], "mileage": [12_000.0, 90_000.0]}
    )
    assert checks.check_consistency(df, FRESH, MAX_FRESH_MILEAGE).passed


def test_consistency_skip_when_columns_absent(clean_df):
    r = checks.check_consistency(
        clean_df.drop(columns=["mileage"]), FRESH, MAX_FRESH_MILEAGE
    )
    assert r.passed
    assert "skipped" in r.detail


# ------------------------------------------------------------------ JSON-safe offender sample


def test_offender_sample_is_json_serializable(dirty_df):
    """Samples (which may hold NaN/Timestamps/numpy types) must survive json.dumps for the report."""
    r = checks.check_ranges(dirty_df, RANGES)
    dumped = json.dumps(asdict(r))
    assert isinstance(r.sample, list) and r.sample
    assert json.loads(dumped)["name"] == "ranges"


def test_offending_index_pinpoints_the_bad_rows(dirty_df):
    """offending_index is the single source of truth the pipeline quarantines on."""
    ranges = checks.check_ranges(dirty_df, RANGES)
    assert set(ranges.offending_index) == {
        0,
        1,
    }  # negative price (row 0), impossible year (row 1)
    dupes = checks.check_duplicates(dirty_df)
    assert set(dupes.offending_index) == {0, 3}  # both rows sharing VIN "V1"
