"""Tests for drift math (PSI) and the run-record round-trip.

`psi()` is implemented, so these run live. The run-store round-trip is skipped until save/load land.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift import psi


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, x) < 1e-6


def test_psi_is_large_for_a_known_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=2000)
    shifted = rng.normal(
        3, 1, size=2000
    )  # clear mean shift -> should breach the 0.2 alert
    assert psi(base, shifted) > 0.2


def test_psi_handles_empty_input():
    assert psi(np.array([]), np.array([1.0, 2.0])) == 0.0


@pytest.mark.skip(
    reason="TODO: implement runs.save_run / load_runs, then round-trip on a temp SQLite DB"
)
def test_run_record_roundtrip(tmp_path):
    """save_run then load_runs returns the same run (SQLite temp engine)."""
    ...
