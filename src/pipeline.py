"""Orchestrate one batch: ingest -> validate(GATE) -> record run -> drift -> db -> features -> train -> evaluate.

`--backfill` replays all time-ordered batches to build the run history the dashboard plots. This
replaces the prototype's single top-level script. The DQ ERROR gate is a HARD stop; drift is a
signal (logged, not fatal).

Run:  python -m src.pipeline [--backfill]
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import Settings, load_settings
from src.db import database
from src.features import preprocess
from src.ingest import dataset
from src.model import evaluate as evaluate_mod
from src.model import train as train_mod
from src.monitoring import runs as runs_mod
from src.quality import report as report_mod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("pipeline")


def run_batch(df, batch_label: str, settings: Settings) -> int:
    """Run the pipeline over one batch. Returns 0 on success, 1 if the DQ gate stops it."""
    engine = database.get_engine()
    runs_table = settings["database"]["runs_table"]
    record = runs_mod.RunRecord.new(batch_label)
    record.n_rows = len(df)

    # 1. Validate -- HARD GATE (pandera schema + business rules)
    dq = report_mod.run_report(df, settings)
    dq.print_summary()
    dq.to_json(settings["paths"]["dq_report"])
    record.dq_pass_rate = sum(r.passed for r in dq.results) / max(len(dq.results), 1)
    record.n_error_checks = len(dq.errors)
    if not dq.passed:
        log.error(
            "[%s] data-quality gate FAILED; recording run and halting", batch_label
        )
        runs_mod.save_run(record, engine, runs_table)
        return 1

    # 2. Drift vs the previous run (a signal, not a gate)
    prev = runs_mod.previous_run(engine, runs_table)
    if prev is not None:
        log.info(
            "[%s] previous run found; compute drift (see monitoring.drift)", batch_label
        )
        # TODO: load the previous batch key columns and call drift.compute_drift(...); log alerts.

    # 3. Load validated data to SQL
    database.write_df(df, settings["database"]["table"], engine)

    # 4. Features -> 5. Train -> 6. Evaluate
    split = preprocess.split_and_encode(
        preprocess.clean(df),
        test_size=settings["split"]["test_size"],
        seed=settings.seed,
    )
    models = train_mod.train(split, settings["model"], seed=settings.seed)
    evaluate_mod.evaluate(models, split, settings["paths"]["metrics"])

    # 7. Record the run (with the winning model metrics)
    # TODO: set record.mae/rmse/r2 from the winning model in `metrics`.
    runs_mod.save_run(record, engine, runs_table)
    log.info("[%s] pipeline complete", batch_label)
    return 0


def run(backfill: bool = False) -> int:
    """Run one batch (the whole file) or replay all time-ordered batches (`--backfill`)."""
    settings = load_settings()
    df = dataset.load(settings)
    log.info("ingested %d rows", len(df))

    if not backfill:
        return run_batch(df, batch_label="all", settings=settings)

    rc = 0
    for label, batch in dataset.iter_batches(df, settings["batching"]):
        rc |= run_batch(batch, batch_label=label, settings=settings)
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
