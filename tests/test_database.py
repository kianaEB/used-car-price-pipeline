"""Tests for SQL write/read round-trip (use a temporary SQLite file, not the real DB)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement database.write_df / read_df (SPEC 9)")
def test_write_read_roundtrip(tmp_path, clean_df):
    """write_df(clean_df, 'cars') then read_df('cars') returns the same rows."""
    ...
