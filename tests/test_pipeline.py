"""End-to-end pipeline tests: quarantine-plus-ceiling gate, backfill history, and drift threading.

Definition-of-Done items: a single batch runs end-to-end and records model metrics; a backfill
yields >=3 runs; a batch-fatal or over-ceiling batch hard-halts before any load/train; quarantine
drops+counts exactly the DQ-flagged ERROR rows and trains on the remainder; and drift is computed
against the previous batch. A temp SQLite engine + a small model config keep it fast and isolated
(sidestepping the machine's locked pytest tmp_path base).
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import Engine, create_engine, inspect

from src import pipeline
from src.config import Settings, load_settings
from src.ingest.dataset import generate_synthetic, iter_batches
from src.monitoring.runs import load_runs
from src.pipeline import run_batch


@contextmanager
def _pipeline_env(
    n_rows: int = 1600, n_weeks: int = 4
) -> Iterator[tuple[Settings, Engine]]:
    """A temp SQLite engine + fast-model settings with all artifact paths redirected to a temp dir."""
    with tempfile.TemporaryDirectory() as directory:
        d = Path(directory)
        engine = create_engine(f"sqlite:///{(d / 'cars.db').as_posix()}")
        settings = load_settings()
        settings.raw["model"]["random_forest"] = {"n_estimators": 12, "max_depth": 8}
        settings.raw["model"]["decision_tree"] = {"max_depth": 8}
        settings.raw["dataset"]["n_rows"] = n_rows
        settings.raw["dataset"]["n_weeks"] = n_weeks
        settings.raw["paths"]["metrics"] = str(d / "metrics.json")
        settings.raw["paths"]["dq_report"] = str(d / "dq_report.json")
        settings.raw["paths"]["model"] = str(d / "model.joblib")
        try:
            yield settings, engine
        finally:
            engine.dispose()


def _runs(engine: Engine, settings: Settings):
    """The runs-table history as a DataFrame."""
    return load_runs(engine, settings["database"]["runs_table"])


# ------------------------------------------------------------------------- happy path + quarantine


def test_single_batch_runs_end_to_end_with_metrics():
    with _pipeline_env() as (settings, engine):
        batch = generate_synthetic(n=800, seed=1, bad_fraction=0.06)
        rc, cleaned = run_batch(batch, "b1", settings, engine)
        runs = _runs(engine, settings)
        cars = inspect(engine).has_table(settings["database"]["table"])
    assert rc == 0
    assert cleaned is not None and 0 < len(cleaned) < len(batch)
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["n_rows"] == 800
    assert row["n_error_checks"] >= 1  # the catch was recorded
    assert row["mae"] > 0  # a model was trained on the clean remainder
    assert cars  # the clean remainder was loaded


def test_quarantine_drops_and_counts_exactly_the_error_rows():
    with _pipeline_env() as (settings, engine):
        batch = generate_synthetic(n=800, seed=1, bad_fraction=0.06)
        defects = batch.attrs["defects"]
        expected_bad = (
            defects["null_brand"]
            + defects["nonpositive_price"]
            + defects["impossible_year"]
            + defects["duplicate_vin"]
        )
        rc, cleaned = run_batch(batch, "b1", settings, engine)
        quarantined = load_runs(engine, settings["database"]["quarantine_table"])
    assert rc == 0
    assert (
        len(batch) - len(cleaned) == expected_bad
    )  # dropped exactly the DQ-flagged rows
    assert len(quarantined) == expected_bad  # ...and audited them
    # the clean remainder carries none of the ERROR defects
    assert (cleaned["price"] > 0).all()
    assert cleaned["year"].between(1950, 2027).all()
    assert cleaned["brand"].notna().all()
    assert (cleaned["vin"] != "DUPLICATE-VIN").all()


# ------------------------------------------------------------------------------- hard-halt gate


def test_batch_fatal_truncated_batch_halts_before_load_and_train():
    with _pipeline_env() as (settings, engine):
        tiny = generate_synthetic(n=50, seed=1, bad_fraction=0.0)  # < min_rows (100)
        rc, cleaned = run_batch(tiny, "tiny", settings, engine)
        runs = _runs(engine, settings)
        loaded = inspect(engine).has_table(settings["database"]["table"])
    assert rc == 1 and cleaned is None
    assert not loaded  # nothing was written to the cars table
    assert len(runs) == 1  # the halted run is still recorded
    assert runs.iloc[0]["mae"] is None or runs.iloc[0].isna()["mae"]  # no model trained


def test_batch_fatal_missing_column_halts():
    with _pipeline_env() as (settings, engine):
        batch = generate_synthetic(n=300, seed=1, bad_fraction=0.0).drop(
            columns=["price"]
        )
        rc, cleaned = run_batch(batch, "no-price", settings, engine)
        loaded = inspect(engine).has_table(settings["database"]["table"])
    assert rc == 1 and cleaned is None
    assert not loaded


def test_over_ceiling_batch_halts_before_load_and_train():
    with _pipeline_env() as (settings, engine):
        batch = generate_synthetic(n=400, seed=1, bad_fraction=0.0)
        batch.loc[:159, "price"] = -1.0  # 40% ERROR rows, above the 25% ceiling
        rc, cleaned = run_batch(batch, "garbage", settings, engine)
        loaded = inspect(engine).has_table(settings["database"]["table"])
        runs = _runs(engine, settings)
    assert rc == 1 and cleaned is None
    assert not loaded
    assert runs.iloc[0].isna()["mae"]  # recorded but not trained


# ---------------------------------------------------------------------------------- backfill


def test_run_all_processes_the_whole_dataset_as_one_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    with _pipeline_env(n_rows=1200, n_weeks=4) as (settings, engine):
        monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
        monkeypatch.setattr(pipeline.database, "get_engine", lambda: engine)
        rc = pipeline.run(backfill=False)
        runs = _runs(engine, settings)
    assert rc == 0
    assert len(runs) == 1
    assert runs.iloc[0]["batch_label"] == "all"
    assert runs.iloc[0]["mae"] > 0


def test_backfill_produces_at_least_three_runs(monkeypatch: pytest.MonkeyPatch):
    with _pipeline_env(n_rows=1600, n_weeks=4) as (settings, engine):
        monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
        monkeypatch.setattr(pipeline.database, "get_engine", lambda: engine)
        rc = pipeline.run(backfill=True)
        runs = _runs(engine, settings)
    assert rc == 0
    assert len(runs) >= 3  # a real run history for the dashboard
    assert runs["mae"].notna().all()  # every batch trained on its clean remainder
    assert runs["n_quarantined"].sum() > 0  # quarantine counts persisted per run
    assert (
        runs["drift"].apply(lambda d: "psi" in d).any()
    )  # drift persisted for the dashboard


# ------------------------------------------------------------------------------------- drift


def test_drift_is_computed_against_the_previous_batch(caplog):
    with _pipeline_env() as (settings, engine):
        drift_cfg = {
            "price_inflation_per_week": 0.1,
            "rising_null_column": "mileage",
            "new_brand_week": 2,
        }
        df = generate_synthetic(
            n=4000, seed=42, bad_fraction=0.0, n_weeks=6, drift=drift_cfg
        )
        batches = list(iter_batches(df, settings["batching"]))
        _, previous_clean = run_batch(batches[0][1], batches[0][0], settings, engine)
        assert previous_clean is not None
        with caplog.at_level(logging.WARNING, logger="pipeline"):
            run_batch(
                batches[-1][1],
                batches[-1][0],
                settings,
                engine,
                previous_clean=previous_clean,
            )
    assert any("DRIFT" in record.getMessage() for record in caplog.records)
