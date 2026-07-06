"""Shared pytest fixtures: a clean dataset and a deliberately 'dirty' one with known defects."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, schema-valid used-car DataFrame with no defects."""
    return pd.DataFrame(
        {
            "price": [12000.0, 8500.0, 30000.0, 15000.0],
            "brand": ["toyota", "ford", "bmw", "honda"],
            "model": ["corolla", "focus", "3 series", "civic"],
            "year": [2018, 2015, 2020, 2019],
            "mileage": [45000.0, 80000.0, 15000.0, 30000.0],
            "title_status": ["clean vehicle"] * 4,
            "vin": ["V1", "V2", "V3", "V4"],
        }
    )


@pytest.fixture
def dirty_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """The clean frame with injected defects, one per DQ check."""
    df = clean_df.copy()
    df.loc[0, "price"] = -5000.0  # negative price   -> check_ranges (ERROR)
    df.loc[1, "year"] = 1200  # impossible year  -> check_ranges (ERROR)
    df.loc[2, "brand"] = None  # null brand       -> check_nulls
    df.loc[3, "vin"] = "V1"  # duplicate VIN     -> check_duplicates
    return df
