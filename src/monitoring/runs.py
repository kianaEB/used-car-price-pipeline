"""Run history for monitoring: persist one RunRecord per pipeline batch to SQL.

The dashboard reads this table to plot quality, drift, and model metrics over time. This is what
turns a one-shot script into something you can *monitor* -- the core of the trivago-style story.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import Engine, inspect


@dataclass
class RunRecord:
    """One pipeline run over one batch of data."""

    run_id: str
    ts: str  # ISO-8601 timestamp (UTC)
    batch_label: str  # e.g. "2021-W03"
    n_rows: int
    dq_pass_rate: float  # fraction of checks that passed
    n_error_checks: int
    freshness_days: float  # age of the newest posting_date at run time
    n_quarantined: int = 0  # ERROR rows dropped by the gate before load/train
    col_stats: dict[str, Any] = field(
        default_factory=dict
    )  # {col: {"mean":..., "null_rate":...}}
    drift: dict[str, Any] = field(
        default_factory=dict
    )  # serialised DriftReport vs the previous run (psi, category_shift, ...)
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None

    @staticmethod
    def new(batch_label: str) -> "RunRecord":
        """Start a RunRecord with a fresh id + UTC timestamp; other fields filled as the run runs."""
        ts = datetime.now(timezone.utc).isoformat()
        return RunRecord(
            run_id=f"{batch_label}-{ts}",
            ts=ts,
            batch_label=batch_label,
            n_rows=0,
            dq_pass_rate=0.0,
            n_error_checks=0,
            freshness_days=0.0,
        )


def column_stats(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, float | None]]:
    """Per-column {mean, null_rate} for the key columns, for the RunRecord and drift comparisons."""
    stats: dict[str, dict[str, float | None]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mean = values.mean()
        stats[col] = {
            "mean": None if pd.isna(mean) else float(mean),
            "null_rate": float(df[col].isna().mean()) if len(df) else 0.0,
        }
    return stats


_JSON_COLUMNS = ("col_stats", "drift")


def save_run(record: RunRecord, engine: Engine, table: str = "runs") -> None:
    """Append one RunRecord to the runs table (dict fields as JSON), creating the table if absent."""
    row = asdict(record)
    for col in _JSON_COLUMNS:
        row[col] = json.dumps(row[col])
    pd.DataFrame([row]).to_sql(table, engine, if_exists="append", index=False)


def load_runs(engine: Engine, table: str = "runs") -> pd.DataFrame:
    """Return the full run history as a DataFrame (empty if the table doesn't exist yet), ts-sorted."""
    if not inspect(engine).has_table(table):
        return pd.DataFrame()
    df = pd.read_sql_table(table, engine)
    for col in _JSON_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda s: json.loads(s) if isinstance(s, str) else {}
            )
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def _row_to_record(row: pd.Series) -> RunRecord:
    """Rebuild a RunRecord from a runs-table row (NaN model metrics -> None)."""

    def _opt(value: Any) -> float | None:
        return None if pd.isna(value) else float(value)

    def _dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    return RunRecord(
        run_id=str(row["run_id"]),
        ts=str(row["ts"]),
        batch_label=str(row["batch_label"]),
        n_rows=int(row["n_rows"]),
        dq_pass_rate=float(row["dq_pass_rate"]),
        n_error_checks=int(row["n_error_checks"]),
        freshness_days=float(row["freshness_days"]),
        n_quarantined=int(row.get("n_quarantined", 0) or 0),
        col_stats=_dict(row.get("col_stats")),
        drift=_dict(row.get("drift")),
        mae=_opt(row.get("mae")),
        rmse=_opt(row.get("rmse")),
        r2=_opt(row.get("r2")),
    )


def latest_run(engine: Engine, table: str = "runs") -> RunRecord | None:
    """Return the most recent run by timestamp, or None if there is no history yet."""
    runs = load_runs(engine, table)
    if runs.empty:
        return None
    return _row_to_record(runs.iloc[-1])


def previous_run(engine: Engine, table: str = "runs") -> RunRecord | None:
    """Return the run to compare the current batch against: the latest one already persisted."""
    return latest_run(engine, table)
