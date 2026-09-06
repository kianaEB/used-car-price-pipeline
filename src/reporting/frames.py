"""Runs-history transforms: the parsed `runs` table -> plot-ready and BI-ready frames.

One parser, two renderers. `dashboard/app.py` renders these with Streamlit and
`src/reporting/bi_export.py` reshapes them into tidy CSVs -- both import from here, so the dashboard
and the BI export can never disagree about what a PSI value or a null-rate means.

Every function is pure and takes the DataFrame `monitoring.runs.load_runs` returns, i.e. with the
`col_stats` and `drift` JSON blobs already parsed to dicts. A value that a run never recorded comes
back as None (NaN in the frame); callers decide whether that is a gap to plot or a row to drop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BATCH_COLUMN = "batch_label"


def quality_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run DQ pass-rate, failed-ERROR-check count, quarantine count, and rows ingested."""
    return runs[
        [BATCH_COLUMN, "dq_pass_rate", "n_error_checks", "n_quarantined", "n_rows"]
    ].copy()


def model_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run model error metrics (MAE/RMSE/R2); NaN on halted runs that never trained."""
    return runs[[BATCH_COLUMN, "mae", "rmse", "r2"]].copy()


def freshness_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run data freshness (age in days of the newest posting_date)."""
    return runs[[BATCH_COLUMN, "freshness_days"]].copy()


def psi_frame(
    runs: pd.DataFrame, columns: list[str], id_columns: list[str] | None = None
) -> pd.DataFrame:
    """Per-run PSI for each numeric column (NaN for the first run / halts with no drift)."""
    return _extract_nested(runs, "drift", "psi", columns, id_columns)


def category_shift_frame(
    runs: pd.DataFrame, columns: list[str], id_columns: list[str] | None = None
) -> pd.DataFrame:
    """Per-run top-category share shift for each categorical column (NaN where no drift)."""
    return _extract_nested(runs, "drift", "category_shift", columns, id_columns)


def null_rate_delta_frame(
    runs: pd.DataFrame,
    columns: list[str] | None = None,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Per-run null-rate change vs the previous run; defaults to every column any run recorded.

    Unlike PSI and category shift, `compute_drift` records this for *every* shared column rather
    than a configured shortlist, so the default column list is the union across the history.
    """
    columns = columns if columns is not None else drift_columns(runs, "null_rate_delta")
    return _extract_nested(runs, "drift", "null_rate_delta", columns, id_columns)


def col_stat_frame(
    runs: pd.DataFrame,
    columns: list[str],
    stat: str,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Per-run value of one col_stats statistic ('mean' or 'null_rate') for each key column."""
    rows = []
    for _, run in runs.iterrows():
        stats = run.get("col_stats") or {}
        entry = _ids(run, id_columns)
        for col in columns:
            entry[col] = (stats.get(col) or {}).get(stat)
        rows.append(entry)
    return pd.DataFrame(rows, columns=[*_id_names(id_columns), *columns])


def null_rate_frame(
    runs: pd.DataFrame, columns: list[str], id_columns: list[str] | None = None
) -> pd.DataFrame:
    """Per-run null fraction for each key column, read from the run's col_stats."""
    return col_stat_frame(runs, columns, "null_rate", id_columns)


def drift_columns(runs: pd.DataFrame, inner: str) -> list[str]:
    """Sorted union of the column names any run recorded under drift[inner]."""
    names: set[str] = set()
    for _, run in runs.iterrows():
        names.update((run.get("drift") or {}).get(inner) or {})
    return sorted(names)


def _extract_nested(
    runs: pd.DataFrame,
    outer: str,
    inner: str,
    columns: list[str],
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Pull runs[outer][inner][col] into a per-run frame (NaN when the mapping is absent)."""
    rows = []
    for _, run in runs.iterrows():
        mapping = (run.get(outer) or {}).get(inner) or {}
        entry = _ids(run, id_columns)
        for col in columns:
            entry[col] = mapping.get(col)
        rows.append(entry)
    return pd.DataFrame(rows, columns=[*_id_names(id_columns), *columns])


def _id_names(id_columns: list[str] | None) -> list[str]:
    """Identifying columns to carry onto each output row; the dashboard's axis unless overridden."""
    return list(id_columns) if id_columns is not None else [BATCH_COLUMN]


def _ids(run: pd.Series, id_columns: list[str] | None) -> dict[str, Any]:
    """The identifying values for one run, as the seed of its output row."""
    return {name: run[name] for name in _id_names(id_columns)}
