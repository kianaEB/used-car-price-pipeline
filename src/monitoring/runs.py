"""Run history for monitoring: persist one RunRecord per pipeline batch to SQL.

The dashboard reads this table to plot quality, drift, and model metrics over time. This is what
turns a one-shot script into something you can *monitor* -- the core of the trivago-style story.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import Engine


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
    col_stats: dict[str, Any] = field(default_factory=dict)  # {col: {"mean":..., "null_rate":...}}
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


def save_run(record: RunRecord, engine: Engine, table: str = "runs") -> None:
    """Append one RunRecord to the runs table (col_stats serialized as JSON).

    TODO: json.dumps(col_stats); build a one-row DataFrame; df.to_sql(table, engine,
    if_exists="append", index=False).
    """
    raise NotImplementedError


def load_runs(engine: Engine, table: str = "runs") -> pd.DataFrame:
    """Return the full run history as a DataFrame (empty if the table doesn't exist yet).

    TODO: read the table; parse col_stats JSON back to dicts; sort by ts ascending.
    """
    raise NotImplementedError


def previous_run(engine: Engine, table: str = "runs") -> RunRecord | None:
    """Return the most recent prior run, or None if this is the first run. TODO."""
    raise NotImplementedError
