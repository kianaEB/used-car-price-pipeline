"""Tests for monitoring: PSI math, the run-history round-trip, and the drift signals.

The Definition-of-Done items live here: psi() ~ 0 for identical distributions and large for a
known shift, save_run/load_runs round-trip, and a known distribution shift raises a drift alert
while identical data does not. Shift cases use the generator's built-in drift (price inflation,
rising null-rate, new brand). A temp-file SQLite engine stands in for the DB (in-memory + disposed
before cleanup sidesteps the machine's locked pytest tmp_path base).
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine

from src.config import load_settings
from src.ingest.dataset import generate_synthetic, iter_batches
from src.monitoring.drift import (
    category_shift,
    compute_drift,
    freshness,
    null_rate_delta,
    psi,
)
from src.monitoring.runs import (
    RunRecord,
    column_stats,
    latest_run,
    load_runs,
    previous_run,
    save_run,
)


@contextmanager
def _engine() -> Iterator[Engine]:
    """A throwaway SQLite engine on a temp file, disposed before the dir is cleaned up."""
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{(Path(directory) / 'runs.db').as_posix()}")
        try:
            yield engine
        finally:
            engine.dispose()


def _record(label: str, ts: str, mae: float | None = 1234.5) -> RunRecord:
    """A fully-populated RunRecord for round-trip tests."""
    return RunRecord(
        run_id=f"{label}-{ts}",
        ts=ts,
        batch_label=label,
        n_rows=500,
        dq_pass_rate=0.86,
        n_error_checks=1,
        freshness_days=5.0,
        col_stats={"price": {"mean": 15000.0, "null_rate": 0.0}},
        mae=mae,
    )


def _monitoring_cfg() -> dict:
    """The monitoring config block (drift thresholds + freshness)."""
    return load_settings()["monitoring"]


def _weekly(min_rows: int = 100) -> dict[str, object]:
    """Weekly batching config."""
    return {"date_column": "posting_date", "freq": "W", "min_batch_rows": min_rows}


# --------------------------------------------------------------------------------- PSI


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


def test_psi_handles_degenerate_bins():
    """A constant expected distribution collapses to <2 bin edges and returns 0.0, not a crash."""
    assert psi(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])) == 0.0


# ------------------------------------------------------------------- run history round-trip


def test_save_and_load_runs_roundtrip():
    with _engine() as engine:
        record = _record("2024-W01", "2024-01-01T00:00:00+00:00")
        save_run(record, engine)
        runs = load_runs(engine)
        assert len(runs) == 1
        assert runs.iloc[0]["batch_label"] == "2024-W01"
        assert runs.iloc[0]["col_stats"] == {
            "price": {"mean": 15000.0, "null_rate": 0.0}
        }
        assert previous_run(engine) == record  # dataclass equality, col_stats included


def test_run_record_new_sets_id_and_timestamp():
    record = RunRecord.new("2024-W05")
    assert record.batch_label == "2024-W05"
    assert record.run_id.startswith("2024-W05-")
    assert record.ts in record.run_id and record.n_rows == 0


def test_none_model_metrics_roundtrip_as_none():
    with _engine() as engine:
        save_run(_record("2024-W02", "2024-01-08T00:00:00+00:00", mae=None), engine)
        loaded = latest_run(engine)
        assert loaded is not None
        assert loaded.mae is None and loaded.rmse is None and loaded.r2 is None


def test_latest_and_previous_return_most_recent():
    with _engine() as engine:
        save_run(_record("2024-W01", "2024-01-01T00:00:00+00:00"), engine)
        save_run(_record("2024-W02", "2024-01-08T00:00:00+00:00"), engine)
        assert latest_run(engine).batch_label == "2024-W02"
        assert previous_run(engine).batch_label == "2024-W02"


def test_load_runs_empty_without_table():
    with _engine() as engine:
        assert load_runs(engine).empty
        assert previous_run(engine) is None
        assert latest_run(engine) is None


def test_column_stats_reports_mean_and_null_rate():
    df = pd.DataFrame({"price": [100.0, 200.0, None], "mileage": [10.0, 20.0, 30.0]})
    stats = column_stats(df, ["price", "mileage", "absent"])
    assert stats["price"]["mean"] == pytest.approx(150.0)
    assert stats["price"]["null_rate"] == pytest.approx(1 / 3)
    assert "absent" not in stats


# ---------------------------------------------------------------------------- drift signals


def test_null_rate_delta_detects_rising_nulls():
    prev = pd.DataFrame({"mileage": [1.0, 2.0, 3.0, 4.0]})
    curr = pd.DataFrame({"mileage": [1.0, None, None, 4.0]})
    assert null_rate_delta(prev, curr)["mileage"] == pytest.approx(0.5)


def test_category_shift_detects_new_category():
    prev = pd.DataFrame({"brand": ["a", "a", "b", "b"]})
    curr = pd.DataFrame({"brand": ["a", "c", "c", "c"]})  # 'c' is new at 75%
    assert category_shift(prev, curr, ["brand"])["brand"] >= 0.5


def test_null_rate_delta_skips_columns_absent_from_previous():
    delta = null_rate_delta(
        pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1], "b": [2]})
    )
    assert "a" in delta and "b" not in delta


def test_category_shift_skips_absent_column():
    assert (
        category_shift(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), ["brand"])
        == {}
    )


def test_freshness_flags_stale_and_passes_fresh():
    fresh = pd.DataFrame({"posting_date": pd.to_datetime(["2024-02-28", "2024-02-20"])})
    stale = pd.DataFrame({"posting_date": pd.to_datetime(["2024-01-01", "2023-12-01"])})
    assert freshness(fresh, "2024-03-01") <= 30
    assert freshness(stale, "2024-03-01") > 30
    assert freshness(fresh, None) == 0.0  # no reference -> no signal
    assert freshness(pd.DataFrame({"x": [1]}), "2024-03-01") == 0.0  # no posting_date
    assert freshness(pd.DataFrame({"posting_date": [pd.NaT]}), "2024-03-01") == 0.0


# ------------------------------------------------------------- Definition-of-Done: alert vs none


def test_compute_drift_flags_a_known_distribution_shift():
    """Early vs late batch of a drifted dataset trips PSI, null-rate, and category alerts."""
    drift = {
        "price_inflation_per_week": 0.05,
        "rising_null_column": "mileage",
        "new_brand_week": 3,
    }
    df = generate_synthetic(n=8000, seed=42, bad_fraction=0.0, n_weeks=8, drift=drift)
    batches = list(iter_batches(df, _weekly()))
    early, late = batches[0][1], batches[-1][1]
    report = compute_drift(early, late, _monitoring_cfg())
    assert report.has_alert
    assert report.psi["price"] > 0.2
    assert any("mileage" in alert for alert in report.alerts)  # rising null-rate
    assert any(
        "brand" in alert for alert in report.alerts
    )  # new brand -> category shift


def test_compute_drift_no_alert_on_identical_data():
    """Comparing a fresh batch to itself yields no drift and no freshness alert."""
    drift = {
        "price_inflation_per_week": 0.02,
        "rising_null_column": "mileage",
        "new_brand_week": 3,
    }
    df = generate_synthetic(n=4000, seed=42, bad_fraction=0.0, n_weeks=8, drift=drift)
    late = list(iter_batches(df, _weekly()))[-1][1]
    report = compute_drift(late, late, _monitoring_cfg())
    assert not report.has_alert
    assert report.psi["price"] < 1e-6


def test_price_shock_trips_psi_once_across_consecutive_weeks():
    """The injected one-time price shock trips PSI at exactly one consecutive-week transition."""
    drift = {  # isolate the shock: gradual inflation stays below the alert, no other signals
        "price_inflation_per_week": 0.015,
        "price_shock_week": 4,
        "price_shock_multiplier": 1.5,
    }
    df = generate_synthetic(n=8000, seed=42, bad_fraction=0.0, n_weeks=8, drift=drift)
    batches = [batch for _, batch in iter_batches(df, _weekly())]
    cfg = _monitoring_cfg()
    alert = cfg["drift"]["psi_alert"]
    price_psi = [
        compute_drift(prev, curr, cfg).psi["price"]
        for prev, curr in zip(batches, batches[1:])
    ]
    tripped = [value > alert for value in price_psi]
    assert sum(tripped) == 1  # exactly one transition trips PSI...
    assert tripped[
        3
    ]  # ...the W04->W05 shock (4th consecutive pair); gradual weeks stay below


def test_compute_drift_flags_a_stale_batch():
    """An early (old) batch trips the freshness alert even when its distribution hasn't drifted."""
    df = generate_synthetic(n=4000, seed=42, bad_fraction=0.0, n_weeks=8)
    early = list(iter_batches(df, _weekly()))[0][1]
    report = compute_drift(early, early, _monitoring_cfg())
    assert report.has_alert
    assert any("stale" in alert for alert in report.alerts)
    assert report.psi["price"] < 1e-6  # only freshness fired, not distribution drift
