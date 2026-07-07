"""Orchestrate one batch: ingest -> validate(GATE) -> record run -> drift -> db -> features -> train -> evaluate.

The gate is quarantine-plus-ceiling: a batch-fatal problem (too few rows, or a missing required
column) or an ERROR-row fraction above quality.max_error_fraction HARD-HALTS the batch (the run is
recorded as halted, nothing is loaded or trained). Otherwise the ERROR rows -- exactly the rows the
DQ layer flagged, never re-derived -- are quarantined (dropped, counted, audited) and the pipeline
proceeds on the validated-clean remainder. Drift is a signal, logged but never fatal.

`--backfill` replays all time-ordered batches to build the run history the dashboard plots, threading
the previous cleaned batch forward so drift/PSI compares against the real previous distribution.

Run:  python -m src.pipeline [--backfill]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict

import pandas as pd
from sqlalchemy import Engine

from src.config import Settings, load_settings
from src.db import database
from src.features import preprocess
from src.ingest import dataset
from src.model import evaluate as evaluate_mod
from src.model import train as train_mod
from src.monitoring import drift as drift_mod
from src.monitoring import runs as runs_mod
from src.quality import report as report_mod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("pipeline")

# ERROR checks that condemn the whole batch (no per-row quarantine can rescue them).
BATCH_FATAL_CHECKS = {"columns", "min_rows"}


def _error_row_index(report: report_mod.DataQualityReport) -> set[int]:
    """Union of offending row indices across all failed row-level ERROR checks (the DQ verdict)."""
    rows: set[int] = set()
    for result in report.errors:
        if result.name not in BATCH_FATAL_CHECKS:
            rows.update(result.offending_index)
    return rows


def _is_batch_fatal(report: report_mod.DataQualityReport) -> bool:
    """True if a failed ERROR check condemns the batch (empty/truncated or missing columns)."""
    return any(result.name in BATCH_FATAL_CHECKS for result in report.errors)


def _write_quarantine(
    bad_rows: pd.DataFrame, settings: Settings, engine: Engine
) -> None:
    """Append the dropped ERROR rows to the quarantine table for audit (if one is configured)."""
    table = settings["database"].get("quarantine_table")
    if table and len(bad_rows):
        database.write_df(bad_rows, table, engine, if_exists="append")


def run_batch(
    df: pd.DataFrame,
    batch_label: str,
    settings: Settings,
    engine: Engine,
    previous_clean: pd.DataFrame | None = None,
) -> tuple[int, pd.DataFrame | None]:
    """Run one batch end-to-end; return (0, clean_df) on success or (1, None) if the gate halts."""
    quality = settings["quality"]
    runs_table = settings["database"]["runs_table"]
    record = runs_mod.RunRecord.new(batch_label)
    record.n_rows = len(df)

    # 1. Data-quality report -- record the catch regardless of the gate outcome.
    dq = report_mod.run_report(df, settings)
    dq.print_summary()
    dq.to_json(settings.path("dq_report"))
    record.dq_pass_rate = sum(r.passed for r in dq.results) / max(len(dq.results), 1)
    record.n_error_checks = len(dq.errors)

    # 2. Gate: quarantine-plus-ceiling.
    error_rows = _error_row_index(dq)
    error_fraction = len(error_rows) / max(len(df), 1)
    ceiling = quality["max_error_fraction"]
    if _is_batch_fatal(dq) or error_fraction > ceiling:
        reason = (
            "batch-fatal (empty/truncated or missing columns)"
            if _is_batch_fatal(dq)
            else f"ERROR-row fraction {error_fraction:.1%} > ceiling {ceiling:.0%}"
        )
        log.error(
            "[%s] HARD HALT: %s; recording halted run, no load/train",
            batch_label,
            reason,
        )
        runs_mod.save_run(record, engine, runs_table)
        return 1, None

    # Quarantine: drop exactly the DQ-flagged ERROR rows, audit them, proceed on the remainder.
    clean_df = df.drop(index=list(error_rows)).reset_index(drop=True)
    record.n_quarantined = len(error_rows)
    if error_rows:
        log.warning(
            "[%s] quarantined %d ERROR row(s) (%.1f%%); proceeding on %d clean rows",
            batch_label,
            len(error_rows),
            error_fraction * 100,
            len(clean_df),
        )
        _write_quarantine(df.loc[list(error_rows)], settings, engine)

    # 3. Monitoring: per-key-column stats + freshness on the validated-clean remainder.
    reference_date = settings["monitoring"]["freshness"].get("reference_date")
    record.freshness_days = drift_mod.freshness(clean_df, reference_date)
    record.col_stats = runs_mod.column_stats(
        clean_df, settings["monitoring"]["drift"]["psi_columns"]
    )

    # 4. Drift vs the previous cleaned batch (a signal, never a gate); persist it for the dashboard.
    if previous_clean is not None:
        drift_report = drift_mod.compute_drift(
            previous_clean, clean_df, settings["monitoring"]
        )
        record.drift = asdict(drift_report)
        for alert in drift_report.alerts:
            log.warning("[%s] DRIFT: %s", batch_label, alert)

    # 5. Load the clean remainder to SQL.
    database.write_df(
        clean_df, settings["database"]["table"], engine, if_exists="append"
    )

    # 6. Features -> train -> evaluate.
    split = preprocess.split_and_encode(
        preprocess.clean(clean_df, settings),
        test_size=settings["split"]["test_size"],
        seed=settings.seed,
        max_categories=settings["features"].get("max_categories"),
    )
    models = train_mod.train(
        split,
        settings["model"],
        seed=settings.seed,
        artifact_path=settings.path("model"),
    )
    metrics = evaluate_mod.evaluate(models, split, settings.path("metrics"))

    # 7. Record the run with the winning model's metrics.
    best = evaluate_mod.winner(metrics)
    record.mae = metrics[best]["mae"]
    record.rmse = metrics[best]["rmse"]
    record.r2 = metrics[best]["r2"]
    runs_mod.save_run(record, engine, runs_table)
    log.info(
        "[%s] complete: winner=%s MAE=%.2f (%d rows)",
        batch_label,
        best,
        record.mae,
        len(clean_df),
    )
    return 0, clean_df


def _reset_history(engine: Engine, settings: Settings) -> None:
    """Drop the cars/runs/quarantine tables so a run starts from a clean, reproducible history."""
    db = settings["database"]
    tables = [db["table"], db["runs_table"], db.get("quarantine_table")]
    with engine.begin() as conn:
        for table in tables:
            if table:
                conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{table}"')


def run(backfill: bool = False) -> int:
    """Run one batch (the whole file) or replay all time-ordered batches (`--backfill`)."""
    settings = load_settings()
    engine = database.get_engine()
    df = dataset.load(settings)
    log.info("ingested %d rows", len(df))
    _reset_history(engine, settings)

    if not backfill:
        rc, _ = run_batch(df, "all", settings, engine)
        return rc

    rc = 0
    n_runs = 0
    previous_clean: pd.DataFrame | None = None
    for label, batch in dataset.iter_batches(df, settings["batching"]):
        code, cleaned = run_batch(batch, label, settings, engine, previous_clean)
        rc |= code
        n_runs += 1
        if cleaned is not None:
            previous_clean = (
                cleaned  # thread forward so drift sees the real previous distribution
            )
    log.info("backfill complete: %d batch run(s)", n_runs)
    return rc


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run the used-car data pipeline.")
    parser.add_argument(
        "--backfill", action="store_true", help="replay all time-ordered batches"
    )
    args = parser.parse_args()
    return run(backfill=args.backfill)


if __name__ == "__main__":
    sys.exit(_cli())
