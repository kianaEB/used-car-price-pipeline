"""Streamlit monitoring dashboard over the runs table: quality, drift, and model metrics over runs.

The data-shaping helpers (runs history -> plot-ready frames) are pure and unit-tested; the Streamlit
rendering lives only in main(). Importing this module has no side effects except the project-root
sys.path bootstrap below -- the one intended exception, required only so `streamlit run` can find
`src`. The DB connection and column lists come from config (no hardcoded paths).

Run:  streamlit run dashboard/app.py     (or `make dashboard`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Entry-point bootstrap (the intended exception to this module's no-import-side-effects rule):
# `streamlit run dashboard/app.py` puts this file's own folder on sys.path, NOT the repo root, so a
# bare launch can't import `src`. Prepend the project root so every launcher resolves the same way
# (pytest and `python -m` already have the root on the path; make dashboard and bare streamlit need
# this). Must run before the src imports below, hence their E402 exemption.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, load_settings  # noqa: E402
from src.db import database  # noqa: E402
from src.monitoring.runs import load_runs  # noqa: E402

_BATCH = "batch_label"
_BATCH_FATAL = {"columns", "min_rows"}  # ERROR checks that halt rather than quarantine


# --------------------------------------------------------------- pure data-shaping helpers


def quality_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run DQ pass-rate, failed-ERROR-check count, quarantine count, and rows ingested."""
    return runs[
        [_BATCH, "dq_pass_rate", "n_error_checks", "n_quarantined", "n_rows"]
    ].copy()


def model_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run model error metrics (MAE/RMSE/R2); NaN on halted runs that never trained."""
    return runs[[_BATCH, "mae", "rmse", "r2"]].copy()


def freshness_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run data freshness (age in days of the newest posting_date)."""
    return runs[[_BATCH, "freshness_days"]].copy()


def psi_frame(runs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-run PSI for each numeric column (NaN for the first run / halts with no drift)."""
    return _extract_nested(runs, "drift", "psi", columns)


def category_shift_frame(runs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-run top-category share shift for each categorical column (NaN where no drift)."""
    return _extract_nested(runs, "drift", "category_shift", columns)


def null_rate_frame(runs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-run null fraction for each key column, read from the run's col_stats."""
    rows = []
    for _, run in runs.iterrows():
        stats = run.get("col_stats") or {}
        entry: dict[str, Any] = {_BATCH: run[_BATCH]}
        for col in columns:
            entry[col] = (stats.get(col) or {}).get("null_rate")
        rows.append(entry)
    return pd.DataFrame(rows)


def _extract_nested(
    runs: pd.DataFrame, outer: str, inner: str, columns: list[str]
) -> pd.DataFrame:
    """Pull runs[outer][inner][col] into a per-run frame (NaN when the mapping is absent)."""
    rows = []
    for _, run in runs.iterrows():
        mapping = (run.get(outer) or {}).get(inner) or {}
        entry: dict[str, Any] = {_BATCH: run[_BATCH]}
        for col in columns:
            entry[col] = mapping.get(col)
        rows.append(entry)
    return pd.DataFrame(rows)


def report_quarantine_count(report: list[dict]) -> int:
    """Rows the latest DQ report would quarantine: union of offending rows over row-level ERROR checks."""
    bad: set[int] = set()
    for check in report:
        row_level_error = (
            check.get("severity") == "ERROR"
            and not check.get("passed")
            and check.get("name") not in _BATCH_FATAL
        )
        if row_level_error:
            bad.update(check.get("offending_index", []))
    return len(bad)


# ------------------------------------------------------------------ config-driven data access


def load_history(settings: Settings | None = None) -> pd.DataFrame:
    """Load the run history from the config-driven DB (empty frame if there is none yet)."""
    settings = settings if settings is not None else load_settings()
    return load_runs(database.get_engine(), settings["database"]["runs_table"])


def latest_report(settings: Settings | None = None) -> list[dict]:
    """Load the latest DQ report JSON (empty list if it hasn't been written yet)."""
    settings = settings if settings is not None else load_settings()
    path = settings.path("dq_report")
    return json.loads(path.read_text()) if path.exists() else []


# ----------------------------------------------------------------------- Streamlit entry point


def main() -> None:
    """Render the Streamlit dashboard -- the only entry point with side effects."""
    import streamlit as st

    settings = load_settings()
    drift_cfg = settings["monitoring"]["drift"]
    psi_columns = drift_cfg["psi_columns"]
    category_columns = drift_cfg["category_columns"]

    st.set_page_config(
        page_title="Used-Car Pipeline -- DQ & Drift Monitor", layout="wide"
    )
    st.title("Used-Car Pipeline -- Data Quality & Drift Monitor")
    st.caption(
        "Quality, drift, and model metrics over time-ordered batches (from the runs table)."
    )

    runs = load_history(settings)
    if runs.empty:
        st.warning(
            "The runs table is empty. Run `make backfill` to build the run history."
        )
        return
    report = latest_report(settings)

    trained = runs["mae"].notna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs recorded", len(runs))
    c2.metric("Latest DQ pass rate", f"{runs['dq_pass_rate'].iloc[-1]:.0%}")
    c3.metric("Latest MAE", f"{runs['mae'].iloc[-1]:,.0f}" if trained.any() else "n/a")
    c4.metric("Quarantined (latest report)", report_quarantine_count(report))

    quality = quality_frame(runs).set_index(_BATCH)
    st.subheader("Data-quality pass rate over runs")
    st.line_chart(quality["dq_pass_rate"])

    st.subheader("Rows ingested vs quarantined per run")
    st.bar_chart(quality[["n_rows", "n_quarantined"]])

    st.subheader("Population Stability Index (PSI) per run")
    st.caption(
        f"Alert threshold {drift_cfg['psi_alert']}: a value above it flags price/mileage/year drift."
    )
    st.line_chart(psi_frame(runs, psi_columns).set_index(_BATCH))

    st.subheader("Null-rate per key column over runs")
    st.line_chart(null_rate_frame(runs, psi_columns).set_index(_BATCH))

    st.subheader("Category-share shift per run")
    st.line_chart(category_shift_frame(runs, category_columns).set_index(_BATCH))

    st.subheader("Data freshness (days) per run")
    st.line_chart(freshness_frame(runs).set_index(_BATCH))

    if trained.any():
        st.subheader("Model error (MAE / RMSE) over runs")
        st.line_chart(model_frame(runs).set_index(_BATCH)[["mae", "rmse"]])

    st.subheader("Latest data-quality report")
    if report:
        columns = ["name", "passed", "severity", "n_violations", "detail"]
        st.dataframe(pd.DataFrame(report)[columns], width="stretch")
    else:
        st.info("No dq_report.json yet -- run the pipeline first.")

    st.subheader("Run history")
    st.dataframe(runs, width="stretch")


# Streamlit execs this file as "__main__"; the guard keeps importing it (e.g. in tests) side-effect-free.
if __name__ == "__main__":
    main()
