"""Tests for the pandera technical schema layer (Layer 1 of data quality).

Skipped automatically if pandera isn't installed, so the offline test subset stays green.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandera")

from src.quality.schema import validate_schema  # noqa: E402

RANGES = {"price": [1, 1_000_000], "year": [1950, 2027], "mileage": [0, 1_000_000]}


def test_schema_accepts_clean(clean_df):
    assert validate_schema(clean_df, RANGES).passed


def test_schema_flags_out_of_range(dirty_df):
    r = validate_schema(dirty_df, RANGES)
    assert not r.passed
    assert r.severity == "ERROR"
    assert r.n_violations >= 1


def test_schema_offending_index_marks_failing_rows(dirty_df):
    r = validate_schema(dirty_df, RANGES)
    # negative price (row 0), impossible year (row 1), null brand (row 2); the VIN dup (row 3) is
    # not a schema concern.
    assert {0, 1, 2} <= set(r.offending_index)
    assert 3 not in r.offending_index
