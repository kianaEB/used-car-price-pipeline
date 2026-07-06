"""Tests for the dashboard's data-shaping helpers (runs history -> plot-ready frames).

The Streamlit rendering is not unit-tested; these cover the pure transforms against a small fixture
runs table (the shape load_runs returns: col_stats and drift already parsed to dicts) plus the
latest-report quarantine count and the config-driven history loader.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from dashboard.app import (
    category_shift_frame,
    freshness_frame,
    latest_report,
    load_history,
    model_frame,
    null_rate_frame,
    psi_frame,
    quality_frame,
    report_quarantine_count,
)
from src.config import load_settings
from src.monitoring.runs import RunRecord, save_run

PSI_COLS = ["price", "mileage", "year"]
CAT_COLS = ["title_status", "brand"]


def _runs() -> pd.DataFrame:
    """A three-run fixture matching load_runs' output (first run has no drift yet)."""
    return pd.DataFrame(
        [
            {
                "batch_label": "2024-W01",
                "n_rows": 1000,
                "dq_pass_rate": 0.50,
                "n_error_checks": 3,
                "n_quarantined": 40,
                "freshness_days": 54.0,
                "mae": 900.0,
                "rmse": 1300.0,
                "r2": 0.96,
                "col_stats": {
                    "price": {"mean": 8000.0, "null_rate": 0.0},
                    "mileage": {"mean": 50000.0, "null_rate": 0.02},
                    "year": {"mean": 2015.0, "null_rate": 0.0},
                },
                "drift": {},  # first run has no previous batch to compare against
            },
            {
                "batch_label": "2024-W02",
                "n_rows": 1020,
                "dq_pass_rate": 0.50,
                "n_error_checks": 3,
                "n_quarantined": 55,
                "freshness_days": 47.0,
                "mae": 1000.0,
                "rmse": 1400.0,
                "r2": 0.95,
                "col_stats": {
                    "price": {"mean": 8100.0, "null_rate": 0.0},
                    "mileage": {"mean": 51000.0, "null_rate": 0.05},
                    "year": {"mean": 2015.0, "null_rate": 0.0},
                },
                "drift": {
                    "psi": {"price": 0.03, "mileage": 0.02, "year": 0.02},
                    "category_shift": {"title_status": 0.01, "brand": 0.02},
                    "alerts": [],
                },
            },
            {
                "batch_label": "2024-W05",
                "n_rows": 1010,
                "dq_pass_rate": 0.375,
                "n_error_checks": 3,
                "n_quarantined": 49,
                "freshness_days": 26.0,
                "mae": 1150.0,
                "rmse": 1600.0,
                "r2": 0.95,
                "col_stats": {
                    "price": {"mean": 12000.0, "null_rate": 0.0},
                    "mileage": {"mean": 50000.0, "null_rate": 0.15},
                    "year": {"mean": 2015.0, "null_rate": 0.0},
                },
                "drift": {
                    "psi": {"price": 0.35, "mileage": 0.01, "year": 0.01},
                    "category_shift": {"brand": 0.05, "title_status": 0.01},
                    "alerts": ["PSI drift on price: 0.350 > 0.2"],
                },
            },
        ]
    )


def test_quality_frame_carries_passrate_quarantine_and_rows():
    q = quality_frame(_runs())
    assert list(q.columns) == [
        "batch_label",
        "dq_pass_rate",
        "n_error_checks",
        "n_quarantined",
        "n_rows",
    ]
    assert q["n_quarantined"].tolist() == [40, 55, 49]
    assert q["dq_pass_rate"].iloc[-1] == 0.375


def test_model_frame_exposes_metrics_per_run():
    m = model_frame(_runs())
    assert m["mae"].tolist() == [900.0, 1000.0, 1150.0]
    assert list(m.columns) == ["batch_label", "mae", "rmse", "r2"]


def test_freshness_frame_tracks_days():
    f = freshness_frame(_runs())
    assert f["freshness_days"].tolist() == [54.0, 47.0, 26.0]


def test_psi_frame_reads_drift_and_is_nan_for_first_run():
    p = psi_frame(_runs(), PSI_COLS)
    assert list(p.columns) == ["batch_label", "price", "mileage", "year"]
    assert np.isnan(p["price"].iloc[0])  # first run has no drift
    assert p["price"].iloc[1] == 0.03
    assert p["price"].iloc[2] == 0.35  # the shock week trips the alert threshold


def test_null_rate_frame_reads_col_stats():
    n = null_rate_frame(_runs(), PSI_COLS)
    assert n["mileage"].tolist() == [0.02, 0.05, 0.15]  # rising null-rate over runs
    assert n["price"].tolist() == [0.0, 0.0, 0.0]


def test_category_shift_frame_reads_drift():
    c = category_shift_frame(_runs(), CAT_COLS)
    assert np.isnan(c["brand"].iloc[0])  # no drift on the first run
    assert c["brand"].iloc[1] == 0.02
    assert c["brand"].iloc[2] == 0.05


def test_report_quarantine_count_unions_row_level_error_offenders():
    report = [
        {"name": "columns", "passed": True, "severity": "ERROR", "offending_index": []},
        {
            "name": "schema",
            "passed": False,
            "severity": "ERROR",
            "offending_index": [0, 1, 2],
        },
        {
            "name": "min_rows",
            "passed": True,
            "severity": "ERROR",
            "offending_index": [],
        },
        {
            "name": "ranges",
            "passed": False,
            "severity": "ERROR",
            "offending_index": [1, 2],
        },
        {
            "name": "duplicates",
            "passed": False,
            "severity": "ERROR",
            "offending_index": [7],
        },
        {
            "name": "consistency",
            "passed": False,
            "severity": "WARN",
            "offending_index": [9],
        },
    ]
    # union of ERROR row-level offenders {0,1,2} | {1,2} | {7}; WARN and batch-fatal excluded
    assert report_quarantine_count(report) == 4


def test_report_quarantine_count_zero_for_clean_report():
    report = [
        {"name": "schema", "passed": True, "severity": "ERROR", "offending_index": []}
    ]
    assert report_quarantine_count(report) == 0


def test_latest_report_reads_json_or_empty(monkeypatch):
    settings = load_settings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dq_report.json"
        settings.raw["paths"]["dq_report"] = str(path)
        assert latest_report(settings) == []  # absent -> empty
        path.write_text(json.dumps([{"name": "schema", "passed": True}]))
        assert latest_report(settings) == [{"name": "schema", "passed": True}]


def test_load_history_reads_the_configured_runs_table(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{(Path(directory) / 'runs.db').as_posix()}")
        settings = load_settings()
        monkeypatch.setattr("dashboard.app.database.get_engine", lambda: engine)
        record = RunRecord.new("2024-W01")
        record.drift = {"psi": {"price": 0.3}}
        save_run(record, engine, settings["database"]["runs_table"])
        history = load_history(settings)
        engine.dispose()
    assert len(history) == 1
    assert history.iloc[0]["batch_label"] == "2024-W01"
    assert history.iloc[0]["drift"] == {
        "psi": {"price": 0.3}
    }  # JSON round-trips to a dict
