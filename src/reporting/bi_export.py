"""Tidy CSV export of the run history, for a BI tool (Power BI) to read instead of the database.

`data/processed/cars.db` is ~1.6 GB of listings and quarantined rows; the reporting grain is the
*run*, and there are a handful of those. So this module reshapes the `runs` table into small,
long-form CSVs under `paths.bi_dir` and ships a generated data dictionary beside them -- because
three different "rates" live in that table with three different denominators, and two of them read
like contradictions if a report puts them side by side unlabelled.

All reshaping goes through `src/reporting/frames.py`, the same transforms the Streamlit dashboard
renders, so the export and the dashboard cannot diverge. Nothing here recomputes a metric: every
value is either projected from a run the pipeline recorded or derived from two such values.

Run:  python -m src.reporting.bi_export     (or `make bi-export`)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from src.config import Settings, load_settings
from src.db import database
from src.monitoring.runs import load_runs
from src.reporting.frames import (
    BATCH_COLUMN,
    category_shift_frame,
    col_stat_frame,
    null_rate_delta_frame,
    psi_frame,
)

log = logging.getLogger("reporting.bi_export")

# Identifying columns carried onto every long-form row. run_id is the key BI relationships use;
# batch_label rides along only so a CSV is readable on its own.
ID_COLUMNS = ["run_id", BATCH_COLUMN]

RUNS_COLUMNS = (
    "run_id",
    "run_seq",
    "ts",
    "batch_label",
    "is_latest_for_batch",
    "n_rows",
    "n_quarantined",
    "n_validated",
    "quarantine_rate",
    "row_pass_rate",
    "dq_check_pass_rate",
    "n_failed_error_checks",
    "freshness_days",
    "mae",
    "rmse",
    "r2",
    "n_drift_alerts",
    "drift_status",
)
QUALITY_COLUMNS = ("run_id", "batch_label", "column", "metric", "value")
DRIFT_COLUMNS = (
    "run_id",
    "batch_label",
    "metric",
    "column",
    "value",
    "threshold",
    "is_alert",
)
ALERT_COLUMNS = ("run_id", "batch_label", "alert")
MODEL_COLUMNS = ("model", "metric", "value", "is_winner", "beats_baseline")
DICTIONARY_COLUMNS = ("table", "column", "source", "grain", "denominator", "definition")
MANIFEST_COLUMNS = (
    "exported_at",
    "source_table",
    "n_runs",
    "first_batch",
    "last_batch",
)

# Why a run's drift blob can be empty. Distinguishing these is the point: a blank in a BI visual
# must never be ambiguous between "not defined", "not reached" and "lost in transit".
BASELINE, COMPUTED, HALTED = "baseline", "computed", "halted"

# The two col_stats statistics the pipeline records per key column.
COL_STATS = ("mean", "null_rate")

# Threshold config key per drift metric, with the same fallbacks compute_drift itself applies
# (src/monitoring/drift.py:110-112), so an absent key cannot make the export and the recorded
# alert list disagree.
_THRESHOLDS: dict[str, tuple[str, float]] = {
    "psi": ("psi_alert", 0.2),
    "null_rate_delta": ("null_rate_delta_alert", 0.10),
    "category_shift": ("category_shift_alert", 0.15),
}


# ----------------------------------------------------------------------------- table builders


def runs_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run: the BI anchor, with derived row-level rates and an explicit drift_status.

    Scalar columns are projected straight from the runs table -- no JSON is parsed here, so nothing
    in this function can diverge from the dashboard. Only the renames (which carry the denominator
    into the name) and the derived columns are new; `data_dictionary()` documents every one.
    """
    if runs.empty:
        return _empty(RUNS_COLUMNS)
    n_rows, n_quarantined = runs["n_rows"], runs["n_quarantined"]
    n_validated = n_rows - n_quarantined
    out = pd.DataFrame(
        {
            "run_id": runs["run_id"].to_numpy(),
            "run_seq": range(
                1, len(runs) + 1
            ),  # load_runs returns the history ts-sorted
            "ts": runs["ts"].to_numpy(),
            "batch_label": runs[BATCH_COLUMN].to_numpy(),
            "is_latest_for_batch": _is_latest_for_batch(runs),
            "n_rows": n_rows.to_numpy(),
            "n_quarantined": n_quarantined.to_numpy(),
            "n_validated": n_validated.to_numpy(),
            "quarantine_rate": _rate(n_quarantined, n_rows),
            "row_pass_rate": _rate(n_validated, n_rows),
            "dq_check_pass_rate": runs["dq_pass_rate"].to_numpy(),
            "n_failed_error_checks": runs["n_error_checks"].to_numpy(),
            "freshness_days": runs["freshness_days"].to_numpy(),
            "mae": runs["mae"].to_numpy(),
            "rmse": runs["rmse"].to_numpy(),
            "r2": runs["r2"].to_numpy(),
            "n_drift_alerts": [len(_alerts(row)) for _, row in runs.iterrows()],
            "drift_status": [_drift_status(row) for _, row in runs.iterrows()],
        }
    )
    return out[list(RUNS_COLUMNS)]


def quality_long(runs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Long form (run x column x metric) of the per-column col_stats: mean and null_rate."""
    if runs.empty:
        return _empty(QUALITY_COLUMNS)
    frames = [
        _melt(col_stat_frame(runs, columns, stat, id_columns=ID_COLUMNS), stat)
        for stat in COL_STATS
    ]
    return _combine(frames, QUALITY_COLUMNS)


def drift_long(runs: pd.DataFrame, drift_cfg: dict[str, Any]) -> pd.DataFrame:
    """Long form (run x metric x column) of every drift signal, with its configured threshold.

    A run with nothing measured -- the first batch, or one the gate halted -- contributes NO rows
    at all rather than a row of nulls; `runs.drift_status` records which case it was. So `value` is
    never null here, and a blank in a BI visual always means "not measured".
    """
    if runs.empty:
        return _empty(DRIFT_COLUMNS)
    wide = {
        "psi": psi_frame(runs, drift_cfg.get("psi_columns", []), id_columns=ID_COLUMNS),
        "null_rate_delta": null_rate_delta_frame(runs, id_columns=ID_COLUMNS),
        "category_shift": category_shift_frame(
            runs, drift_cfg.get("category_columns", []), id_columns=ID_COLUMNS
        ),
    }
    frames = []
    for metric, frame in wide.items():
        key, fallback = _THRESHOLDS[metric]
        threshold = float(drift_cfg.get(key, fallback))
        long = _melt(frame, metric)
        long["threshold"] = threshold
        long["is_alert"] = _alert_mask(metric, long["value"], threshold)
        frames.append(long)
    combined = _combine(frames, DRIFT_COLUMNS)
    combined["is_alert"] = combined["is_alert"].astype(bool)
    return combined


def alerts_long(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per drift alert string the pipeline recorded -- the authoritative alert list."""
    if runs.empty:
        return _empty(ALERT_COLUMNS)
    rows = [
        {"run_id": run["run_id"], "batch_label": run[BATCH_COLUMN], "alert": alert}
        for _, run in runs.iterrows()
        for alert in _alerts(run)
    ]
    return pd.DataFrame(rows, columns=list(ALERT_COLUMNS))


def model_comparison(metrics_path: Path) -> pd.DataFrame:
    """Every model's held-out metrics from metrics.json -- the LATEST run only, a different grain.

    This is the one exported table that does not come from the runs table. It has no per-run
    history and joins to nothing, which `data_dictionary()` states so no visual implies otherwise.
    """
    if not metrics_path.is_file():
        log.warning(
            "no metrics at %s; exporting an empty model comparison", metrics_path
        )
        return _empty(MODEL_COLUMNS)
    report = json.loads(metrics_path.read_text())
    best = report.get("winner")
    beats_baseline = bool(report.get("beats_baseline", False))
    rows = [
        {
            "model": name,
            "metric": metric,
            "value": value,
            "is_winner": name == best,
            "beats_baseline": beats_baseline,
        }
        for name, scores in (report.get("models") or {}).items()
        for metric, value in scores.items()
    ]
    return pd.DataFrame(rows, columns=list(MODEL_COLUMNS))


def manifest(runs: pd.DataFrame, source_table: str) -> pd.DataFrame:
    """A one-row provenance table: when this export ran, from where, and over how many runs."""
    labels = [] if runs.empty else runs[BATCH_COLUMN].tolist()
    return pd.DataFrame(
        [
            {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "source_table": source_table,
                "n_runs": len(runs),
                "first_batch": labels[0] if labels else None,
                "last_batch": labels[-1] if labels else None,
            }
        ],
        columns=list(MANIFEST_COLUMNS),
    )


def data_dictionary() -> pd.DataFrame:
    """The generated column reference: every exported column's source, grain and denominator."""
    rows = [
        {
            "table": table,
            "column": column,
            "source": source,
            "grain": grain,
            "denominator": denominator,
            "definition": definition,
        }
        for table, (grain, columns) in _DICTIONARY.items()
        for column, (source, denominator, definition) in columns.items()
    ]
    return pd.DataFrame(rows, columns=list(DICTIONARY_COLUMNS))


# --------------------------------------------------------------------------------- the export


def build_tables(
    runs: pd.DataFrame, settings: Settings, source_table: str
) -> dict[str, pd.DataFrame]:
    """Build every BI table from an already-loaded run history (pure; no I/O beyond metrics.json)."""
    drift_cfg = settings["monitoring"]["drift"]
    return {
        "runs": runs_table(runs),
        "quality_by_column": quality_long(runs, drift_cfg.get("psi_columns", [])),
        "drift_by_column": drift_long(runs, drift_cfg),
        "drift_alerts": alerts_long(runs),
        "model_comparison": model_comparison(settings.path("metrics")),
        "data_dictionary": data_dictionary(),
        "export_manifest": manifest(runs, source_table),
    }


def export_bi_tables(
    engine: Engine | None = None, settings: Settings | None = None
) -> dict[str, Path]:
    """Write every configured BI table to paths.bi_dir; returns {table name: path written}."""
    settings = settings if settings is not None else load_settings()
    engine = engine if engine is not None else database.get_engine()
    source_table = settings["database"]["runs_table"]
    runs = load_runs(engine, source_table)
    if runs.empty:
        log.warning(
            "no run history in %r; exporting header-only tables so the BI refresh still succeeds",
            source_table,
        )

    tables = build_tables(runs, settings, source_table)
    file_names = settings["reporting"]["tables"]
    out_dir = settings.path("bi_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = out_dir / file_names[name]
        frame.to_csv(path, index=False)
        written[name] = path
        log.info("%-18s %5d row(s) -> %s", name, len(frame), path)
    return written


def _cli() -> int:
    """CLI entry: export the run history to BI-ready CSVs and print where they landed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    written = export_bi_tables()
    print(f"BI export: {len(written)} table(s) -> {written['runs'].parent}")
    return 0


# ------------------------------------------------------------------------------------ helpers


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    """A header-only frame, so an empty run history still exports readable, schema-stable files."""
    return pd.DataFrame(columns=list(columns))


def _rate(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Row-level fraction, guarding the zero-row batch (0 rows in -> 0.0, not a division error)."""
    return (numerator / denominator.where(denominator != 0)).fillna(0.0).to_numpy()


def _alerts(run: pd.Series) -> list[str]:
    """The drift alert strings recorded for one run (empty when drift was never computed)."""
    return (run.get("drift") or {}).get("alerts") or []


def _drift_status(run: pd.Series) -> str:
    """Why this run's drift blob is or is not populated: computed, baseline, or halted.

    A halted batch returns at the DQ gate before monitoring, drift or training run, so it is the
    only case with no model error -- which makes `mae` the reliable discriminator. A clean run with
    no drift had no previous batch to compare against: during a backfill that is the first batch,
    and also any batch that follows a halted one (pipeline.run threads `previous_clean` forward).
    """
    if pd.isna(run.get("mae")):
        return HALTED
    return COMPUTED if (run.get("drift") or {}) else BASELINE


def _is_latest_for_batch(runs: pd.DataFrame) -> list[bool]:
    """Flag the newest run per batch_label, so a batch-keyed visual cannot double-count a re-run."""
    labels = list(runs[BATCH_COLUMN])
    newest = {
        label: position for position, label in enumerate(labels)
    }  # ts-sorted: last wins
    if len(newest) < len(labels):
        log.warning(
            "%d run(s) share only %d distinct batch label(s); key BI relationships on run_id and "
            "filter batch-keyed visuals to is_latest_for_batch",
            len(labels),
            len(newest),
        )
    return [position == newest[label] for position, label in enumerate(labels)]


def _melt(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide (one column per data column) -> long, tagged with the metric; unmeasured values dropped."""
    long = wide.melt(id_vars=ID_COLUMNS, var_name="column", value_name="value")
    long["metric"] = metric
    return long.dropna(subset=["value"])


def _alert_mask(metric: str, values: pd.Series, threshold: float) -> pd.Series:
    """Whether each value breaches its threshold, mirroring compute_drift's own comparisons."""
    compared = values.abs() if metric == "null_rate_delta" else values
    return compared > threshold


def _combine(frames: list[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
    """Concatenate per-metric long frames into one table in the declared column order."""
    if not frames:
        return _empty(columns)
    combined = pd.concat(frames, ignore_index=True)
    return combined.reindex(columns=list(columns)).reset_index(drop=True)


# ------------------------------------------------------------------------- the data dictionary
#
# {table: (grain, {column: (source, denominator, definition)})}. This is the answer to the trap
# this module exists to defuse: dq_check_pass_rate, quarantine_rate and row_pass_rate are three
# true numbers about the same run with three different denominators. A test asserts this covers
# every exported column of every exported table exactly once, so it cannot fall behind the schema.

_FK = "Foreign key to runs.run_id -- the key every BI relationship should use."
_DENORM = "Denormalised from runs for readability; relate on run_id, and put runs[batch_label] on axes."

_DICTIONARY: dict[str, tuple[str, dict[str, tuple[str, str, str]]]] = {
    "runs": (
        "one row per pipeline run",
        {
            "run_id": (
                "runs.run_id",
                "",
                "Unique run key: batch label plus the run's UTC timestamp.",
            ),
            "run_seq": (
                "derived",
                "",
                "Position of the run in timestamp order, 1..n. Use as Power BI's Sort-by column for batch_label, which sorts alphabetically otherwise.",
            ),
            "ts": ("runs.ts", "", "UTC ISO-8601 timestamp at which the run started."),
            "batch_label": (
                "runs.batch_label",
                "",
                "Human label for the time slice (e.g. 2021-W13). NOT guaranteed unique -- see is_latest_for_batch.",
            ),
            "is_latest_for_batch": (
                "derived",
                "",
                "True for the newest run carrying this batch_label. Filter batch-keyed visuals on this so a re-run cannot double-count.",
            ),
            "n_rows": (
                "runs.n_rows",
                "",
                "Rows ingested in the batch, BEFORE quarantine. The denominator of both row-level rates.",
            ),
            "n_quarantined": (
                "runs.n_quarantined",
                "",
                "Rows the DQ gate flagged as ERROR and dropped before database load and training.",
            ),
            "n_validated": (
                "derived",
                "",
                "n_rows minus n_quarantined: the rows actually loaded and modelled.",
            ),
            "quarantine_rate": (
                "derived",
                "rows ingested (n_rows)",
                "ROW-level share of ingested rows quarantined. Not comparable with dq_check_pass_rate, which counts checks.",
            ),
            "row_pass_rate": (
                "derived",
                "rows ingested (n_rows)",
                "ROW-level share of ingested rows that survived the gate. Equals 1 - quarantine_rate.",
            ),
            "dq_check_pass_rate": (
                "runs.dq_pass_rate",
                "data-quality checks run",
                "CHECK-level share of DQ checks that passed. It does not describe rows and is NOT 1 - quarantine_rate; label any card with its denominator.",
            ),
            "n_failed_error_checks": (
                "runs.n_error_checks",
                "",
                "Count of ERROR-severity checks that FAILED in this run -- not the number of ERROR checks defined.",
            ),
            "freshness_days": (
                "runs.freshness_days",
                "",
                "Age in days of the newest posting_date versus the configured reference date.",
            ),
            "mae": (
                "runs.mae",
                "",
                "Held-out mean absolute error of the winning model. Blank on a halted run, which never trained.",
            ),
            "rmse": (
                "runs.rmse",
                "",
                "Held-out root mean squared error of the winning model. Blank on a halted run.",
            ),
            "r2": (
                "runs.r2",
                "",
                "Held-out R-squared of the winning model. Blank on a halted run.",
            ),
            "n_drift_alerts": (
                "derived",
                "",
                "Number of drift alerts recorded for this run; the alerts themselves are rows in drift_alerts.",
            ),
            "drift_status": (
                "derived",
                "",
                "Why drift is present or absent: 'computed', 'baseline' (no previous batch to compare against), or 'halted' (the DQ gate stopped the batch before monitoring ran).",
            ),
        },
    ),
    "quality_by_column": (
        "run x column x metric",
        {
            "run_id": ("runs.run_id", "", _FK),
            "batch_label": ("runs.batch_label", "", _DENORM),
            "column": (
                "runs.col_stats",
                "",
                "The data column this statistic describes (the configured monitoring key columns).",
            ),
            "metric": (
                "derived",
                "",
                "Which statistic the row holds: 'mean' or 'null_rate'.",
            ),
            "value": (
                "runs.col_stats",
                "rows in the validated batch (null_rate only)",
                "The statistic's value on the validated remainder of the batch. Never blank: a run that recorded nothing contributes no rows.",
            ),
        },
    ),
    "drift_by_column": (
        "run x metric x column",
        {
            "run_id": ("runs.run_id", "", _FK),
            "batch_label": ("runs.batch_label", "", _DENORM),
            "metric": (
                "derived",
                "",
                "Which drift signal: 'psi', 'null_rate_delta' or 'category_shift'.",
            ),
            "column": ("runs.drift", "", "The data column the signal was measured on."),
            "value": (
                "runs.drift",
                "",
                "The signal's value versus the previous cleaned batch. Never blank: a run with no drift measured contributes no rows -- see runs.drift_status.",
            ),
            "threshold": (
                "config monitoring.drift",
                "",
                "The configured alert threshold for this metric, exported so a visual compares two columns instead of a typed-in constant.",
            ),
            "is_alert": (
                "derived",
                "",
                "Whether value breaches threshold (absolute value for null_rate_delta), mirroring compute_drift. The authoritative alert list is drift_alerts.",
            ),
        },
    ),
    "drift_alerts": (
        "run x alert",
        {
            "run_id": ("runs.run_id", "", _FK),
            "batch_label": ("runs.batch_label", "", _DENORM),
            "alert": (
                "runs.drift.alerts",
                "",
                "One alert string exactly as the pipeline recorded it at run time. Drift is a signal, not a gate: an alert never halted the run.",
            ),
        },
    ),
    "model_comparison": (
        "model x metric (LATEST run only)",
        {
            "model": (
                "metrics.json",
                "",
                "Model name as scored on the held-out split, including the mean baseline.",
            ),
            "metric": ("metrics.json", "", "One of mae, rmse, r2 or mape."),
            "value": (
                "metrics.json",
                "",
                "Score on the held-out test split of the MOST RECENT run only. This table has no per-run history and joins to nothing -- do not relate it to runs.",
            ),
            "is_winner": (
                "metrics.json",
                "",
                "True for the model metrics.json names as the winner (lowest MAE).",
            ),
            "beats_baseline": (
                "metrics.json",
                "",
                "Whether the winner beat the mean baseline; a file-level fact, repeated on every row.",
            ),
        },
    ),
    "data_dictionary": (
        "one row per exported column",
        {
            "table": ("derived", "", "Which exported table the column belongs to."),
            "column": ("derived", "", "The column name as written to the CSV."),
            "source": (
                "derived",
                "",
                "Where the value comes from: a runs-table column, metrics.json, config, or 'derived'.",
            ),
            "grain": ("derived", "", "What one row of that table represents."),
            "denominator": (
                "derived",
                "",
                "For a rate, what it is a share OF. Blank for counts and identifiers.",
            ),
            "definition": ("derived", "", "What the column means."),
        },
    ),
    "export_manifest": (
        "one row per export",
        {
            "exported_at": (
                "derived",
                "",
                "UTC timestamp at which this export ran -- the 'data as of' card on the report.",
            ),
            "source_table": (
                "config database.runs_table",
                "",
                "The run-history table the export was built from.",
            ),
            "n_runs": ("derived", "", "Number of runs in the exported history."),
            "first_batch": (
                "derived",
                "",
                "Batch label of the earliest run in the export.",
            ),
            "last_batch": (
                "derived",
                "",
                "Batch label of the most recent run in the export.",
            ),
        },
    ),
}


if __name__ == "__main__":
    sys.exit(_cli())
