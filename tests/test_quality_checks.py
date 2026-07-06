"""Tests for the data-quality checks: each must pass on clean data and fail on its defect.

The implemented checks (schema, min_rows) have live tests. The rest are skipped with a TODO
reason pointing at the SPEC -- turn each skip into a real test as you implement the check.
"""

from __future__ import annotations

import pytest

from src.quality import checks

REQUIRED = ["price", "brand", "model", "year", "mileage"]
RANGES = {"price": [1, 1_000_000], "year": [1950, 2027], "mileage": [0, 1_000_000]}


def test_schema_passes_on_clean(clean_df):
    assert checks.check_schema(clean_df, REQUIRED).passed


def test_schema_fails_on_missing_column(clean_df):
    r = checks.check_schema(clean_df.drop(columns=["price"]), REQUIRED)
    assert not r.passed
    assert r.severity == "ERROR"
    assert r.n_violations == 1


def test_min_rows_fails_when_too_few(clean_df):
    assert not checks.check_min_rows(clean_df, min_rows=100).passed


def test_min_rows_passes_when_enough(clean_df):
    assert checks.check_min_rows(clean_df, min_rows=1).passed


@pytest.mark.skip(reason="TODO: implement check_ranges (SPEC 7)")
def test_ranges_flag_negative_price_and_bad_year(dirty_df):
    r = checks.check_ranges(dirty_df, RANGES)
    assert not r.passed
    assert r.severity == "ERROR"
    assert r.n_violations >= 2  # negative price + impossible year


@pytest.mark.skip(reason="TODO: implement check_nulls (SPEC 7)")
def test_nulls_flag_missing_brand(dirty_df):
    assert not checks.check_nulls(dirty_df, {"brand": 0.0}).passed


@pytest.mark.skip(reason="TODO: implement check_duplicates (SPEC 7)")
def test_duplicates_flag_repeated_vin(dirty_df):
    r = checks.check_duplicates(dirty_df)
    assert not r.passed
    assert r.n_violations >= 1
