"""Tests for the BI export layer (run history -> tidy CSVs a BI tool reads).

Three invariants carry most of the weight here, because each one is a way a Power BI report could
silently lie:

1. the long tables never contain a null `value` -- a run with nothing measured contributes no rows
   at all, so a blank in a visual always means "not measured";
2. `dq_check_pass_rate` (check-level) and `quarantine_rate` / `row_pass_rate` (row-level) stay
   separate and separately named, because they are three true numbers with three denominators;
3. the generated data dictionary covers every exported column exactly once, so it cannot fall
   behind a schema change.

The fixture is built with real RunRecords and a temp-file SQLite engine -- nothing here reads
data/processed/cars.db, which is git-ignored and would pass locally but fail in CI.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine

import dashboard.app as app
from src.config import Settings, load_settings
from src.monitoring.runs import RunRecord, save_run
from src.reporting import bi_export, frames

PSI_COLS = ["price", "mileage", "year"]
CAT_COLS = ["title_status", "brand"]
LONG_TABLES = (
    "quality_by_column",
    "drift_by_column",
    "drift_alerts",
    "model_comparison",
)


@contextlib.contextmanager
def _engine() -> Iterator[Engine]:
    """A throwaway SQLite engine on a temp file, disposed before the dir is cleaned up."""
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{(Path(directory) / 'runs.db').as_posix()}")
        try:
            yield engine
        finally:
            engine.dispose()


def _stats(mileage_null_rate: float) -> dict[str, dict[str, float]]:
    """Per-column col_stats with a moving mileage null-rate, as the pipeline records them."""
    return {
        "price": {"mean": 8000.0, "null_rate": 0.0},
        "mileage": {"mean": 50000.0, "null_rate": mileage_null_rate},
        "year": {"mean": 2015.0, "null_rate": 0.0},
    }


def _runs() -> pd.DataFrame:
    """A five-run history in the shape load_runs returns (col_stats/drift already dicts).

    Deliberately covers every awkward case: a HALTED run (gate stopped it before monitoring, so no
    col_stats, no drift, no metrics), a BASELINE run (clean, but no previous batch to compare
    against), two COMPUTED runs -- one of which trips two alerts -- and a duplicated batch_label,
    which is what `is_latest_for_batch` exists for.
    """
    return pd.DataFrame(
        [
            {
                "run_id": "2024-W01-halted",
                "ts": "2024-01-08T00:00:00+00:00",
                "batch_label": "2024-W01",
                "n_rows": 800,
                "n_quarantined": 0,
                "dq_pass_rate": 0.25,
                "n_error_checks": 4,
                "freshness_days": 60.0,
                "col_stats": {},
                "drift": {},
                "mae": None,
                "rmse": None,
                "r2": None,
            },
            {
                "run_id": "2024-W02-baseline",
                "ts": "2024-01-15T00:00:00+00:00",
                "batch_label": "2024-W02",
                "n_rows": 1000,
                "n_quarantined": 250,
                "dq_pass_rate": 0.375,
                "n_error_checks": 3,
                "freshness_days": 54.0,
                "col_stats": _stats(0.02),
                "drift": {},  # no previous batch to compare against
                "mae": 900.0,
                "rmse": 1300.0,
                "r2": 0.96,
            },
            {
                "run_id": "2024-W03-computed",
                "ts": "2024-01-22T00:00:00+00:00",
                "batch_label": "2024-W03",
                "n_rows": 1200,
                "n_quarantined": 300,
                "dq_pass_rate": 0.5,
                "n_error_checks": 2,
                "freshness_days": 47.0,
                "col_stats": _stats(0.05),
                "drift": {
                    "psi": {"price": 0.03, "mileage": 0.02, "year": 0.01},
                    "null_rate_delta": {"price": 0.0, "mileage": 0.01},
                    "category_shift": {"title_status": 0.01, "brand": 0.02},
                    "freshness_days": 47.0,
                    "alerts": [],
                },
                "mae": 1000.0,
                "rmse": 1400.0,
                "r2": 0.95,
            },
            {
                "run_id": "2024-W04-computed",
                "ts": "2024-01-29T00:00:00+00:00",
                "batch_label": "2024-W04",
                "n_rows": 1000,
                "n_quarantined": 400,
                "dq_pass_rate": 0.375,
                "n_error_checks": 3,
                "freshness_days": 40.0,
                "col_stats": _stats(0.15),
                "drift": {
                    "psi": {"price": 0.35, "mileage": 0.01, "year": 0.01},
                    "null_rate_delta": {
                        "price": 0.0,
                        "vin": 0.12,
                    },  # note: not a psi column
                    "category_shift": {"title_status": 0.01, "brand": 0.05},
                    "freshness_days": 40.0,
                    "alerts": [
                        "PSI drift on price: 0.350 > 0.2",
                        "null-rate drift on vin: +0.120",
                    ],
                },
                "mae": 1150.0,
                "rmse": 1600.0,
                "r2": 0.95,
            },
            {
                "run_id": "2024-W04-rerun",
                "ts": "2024-02-05T00:00:00+00:00",
                "batch_label": "2024-W04",  # same label re-run: only the newest is "latest"
                "n_rows": 900,
                "n_quarantined": 90,
                "dq_pass_rate": 0.625,
                "n_error_checks": 1,
                "freshness_days": 33.0,
                "col_stats": _stats(0.03),
                "drift": {
                    "psi": {"price": 0.02, "mileage": 0.01, "year": 0.0},
                    "null_rate_delta": {"price": 0.0},
                    "category_shift": {"title_status": 0.0, "brand": 0.01},
                    "freshness_days": 33.0,
                    "alerts": [],
                },
                "mae": 1200.0,
                "rmse": 1700.0,
                "r2": 0.94,
            },
        ]
    )


def _drift_cfg() -> dict:
    """The monitoring.drift config block (psi_alert 0.2, null 0.10, category 0.15)."""
    return load_settings()["monitoring"]["drift"]


def _settings_into(directory: Path, metrics: Path | None = None) -> Settings:
    """Real settings with the BI output dir (and metrics path) redirected into a temp dir."""
    settings = load_settings()
    settings.raw["paths"]["bi_dir"] = str(directory)
    settings.raw["paths"]["metrics"] = str(
        metrics if metrics is not None else directory / "absent-metrics.json"
    )
    return settings


# ------------------------------------------------------------------ runs.csv: the anchor table


def test_runs_table_is_one_row_per_run_with_run_seq_in_ts_order():
    table = bi_export.runs_table(_runs())
    assert list(table.columns) == list(bi_export.RUNS_COLUMNS)
    assert len(table) == 5
    assert table["run_seq"].tolist() == [1, 2, 3, 4, 5]
    assert table["batch_label"].tolist() == [
        "2024-W01",
        "2024-W02",
        "2024-W03",
        "2024-W04",
        "2024-W04",
    ]


def test_drift_status_labels_baseline_halted_and_computed():
    status = bi_export.runs_table(_runs())["drift_status"].tolist()
    # the gate halted W01 before monitoring ran; W02 ran clean but had no previous batch
    assert status == ["halted", "baseline", "computed", "computed", "computed"]


def test_runs_table_derives_validated_rows_and_both_row_level_rates():
    table = bi_export.runs_table(_runs())
    assert table["n_validated"].tolist() == [800, 750, 900, 600, 810]
    assert table["quarantine_rate"].tolist() == [0.0, 0.25, 0.25, 0.4, 0.1]
    assert table["row_pass_rate"].tolist() == [1.0, 0.75, 0.75, 0.6, 0.9]
    # the two row-level rates are complements of each other; the check-level one is not (below)
    assert (table["quarantine_rate"] + table["row_pass_rate"] == 1.0).all()


def test_check_pass_rate_and_quarantine_rate_are_separate_columns():
    """The three-denominators trap: check-level and row-level rates must not be conflated."""
    table = bi_export.runs_table(_runs())
    row = table[table["run_id"] == "2024-W02-baseline"].iloc[0]
    # 0.375 of CHECKS passed while 0.75 of ROWS did -- both true, neither the other's complement
    assert row["dq_check_pass_rate"] == 0.375
    assert row["row_pass_rate"] == 0.75
    assert row["quarantine_rate"] == 0.25
    assert row["dq_check_pass_rate"] != 1 - row["quarantine_rate"]
    # carried through from the runs table verbatim, never recomputed from rows
    assert table["dq_check_pass_rate"].tolist() == _runs()["dq_pass_rate"].tolist()
    assert table["n_failed_error_checks"].tolist() == _runs()["n_error_checks"].tolist()


def test_is_latest_for_batch_flags_only_the_newest_run_per_label():
    table = bi_export.runs_table(_runs())
    assert table["is_latest_for_batch"].tolist() == [True, True, True, False, True]
    latest = table[table["is_latest_for_batch"]]
    assert not latest["batch_label"].duplicated().any()  # safe as a visual's axis


def test_n_drift_alerts_counts_the_recorded_alert_strings():
    assert bi_export.runs_table(_runs())["n_drift_alerts"].tolist() == [0, 0, 0, 2, 0]


# ------------------------------------------------------------------------- the long-form tables


def test_long_tables_have_no_null_values():
    """The headline invariant: a blank in a BI visual means "not measured", never "lost in transit"."""
    settings = load_settings()
    tables = bi_export.build_tables(_runs(), settings, "runs")
    for name in LONG_TABLES:
        assert not tables[name].isna().any().any(), f"{name} leaked a null"


def test_drift_long_omits_runs_with_no_drift_entirely():
    drift = bi_export.drift_long(_runs(), _drift_cfg())
    absent = {"2024-W01-halted", "2024-W02-baseline"}
    assert absent.isdisjoint(set(drift["run_id"]))  # explicit absence, not a row of NaN
    assert set(drift["run_id"]) == {
        "2024-W03-computed",
        "2024-W04-computed",
        "2024-W04-rerun",
    }


def test_drift_long_is_tidy_run_metric_column_value():
    drift = bi_export.drift_long(_runs(), _drift_cfg())
    assert list(drift.columns) == list(bi_export.DRIFT_COLUMNS)
    assert set(drift["metric"]) == {"psi", "null_rate_delta", "category_shift"}
    assert not drift.duplicated(subset=["run_id", "metric", "column"]).any()
    # 3 psi + 2 null-rate + 2 category for W03, likewise W04, and 3+1+2 for the re-run
    assert len(drift) == 20


def test_drift_long_agrees_with_the_dashboard_psi_frame():
    """The export and the dashboard must show the same PSI, cell for cell."""
    runs = _runs()
    wide = app.psi_frame(runs, PSI_COLS)
    long = bi_export.drift_long(runs, _drift_cfg())
    psi = long[long["metric"] == "psi"]
    for position, (_, run) in enumerate(runs.iterrows()):
        for column in PSI_COLS:
            expected = wide.iloc[position][column]
            match = psi[(psi["run_id"] == run["run_id"]) & (psi["column"] == column)]
            if pd.isna(expected):
                assert (
                    match.empty
                )  # the dashboard's NaN becomes an absent row, not a null
            else:
                assert match["value"].iloc[0] == expected


def test_dashboard_and_export_share_one_parser():
    """One parser, two renderers: these must be the same objects, not two copies that can drift."""
    assert app.psi_frame is frames.psi_frame
    assert app.category_shift_frame is frames.category_shift_frame
    assert app.null_rate_frame is frames.null_rate_frame
    assert app.quality_frame is frames.quality_frame
    assert app.model_frame is frames.model_frame
    assert app.freshness_frame is frames.freshness_frame


def test_is_alert_matches_the_recorded_alert_strings():
    """is_alert is derived, so it is cross-checked against the alerts compute_drift actually wrote."""
    runs = _runs()
    drift = bi_export.drift_long(runs, _drift_cfg())
    alerts = bi_export.alerts_long(runs)
    assert drift["is_alert"].sum() == len(alerts) == 2
    flagged = drift[drift["is_alert"]]
    assert set(flagged["run_id"]) == set(alerts["run_id"]) == {"2024-W04-computed"}
    assert set(zip(flagged["metric"], flagged["column"])) == {
        ("psi", "price"),
        ("null_rate_delta", "vin"),  # abs() breach, mirroring compute_drift
    }


def test_thresholds_are_read_from_config():
    settings = load_settings()
    settings.raw["monitoring"]["drift"]["psi_alert"] = 0.02
    drift = bi_export.drift_long(_runs(), settings["monitoring"]["drift"])
    psi = drift[drift["metric"] == "psi"]
    assert set(psi["threshold"]) == {0.02}
    # lowering the threshold must widen the alert set -- nothing is hardcoded
    assert psi["is_alert"].sum() > 1


def test_quality_long_carries_mean_and_null_rate_per_column():
    quality = bi_export.quality_long(_runs(), PSI_COLS)
    assert list(quality.columns) == list(bi_export.QUALITY_COLUMNS)
    assert set(quality["metric"]) == {"mean", "null_rate"}
    assert "2024-W01-halted" not in set(quality["run_id"])  # no col_stats recorded
    assert len(quality) == 4 * len(PSI_COLS) * 2
    mileage_nulls = quality[
        (quality["column"] == "mileage") & (quality["metric"] == "null_rate")
    ]
    assert mileage_nulls["value"].tolist() == [0.02, 0.05, 0.15, 0.03]


def test_quality_long_agrees_with_the_dashboard_null_rate_frame():
    runs = _runs()
    wide = app.null_rate_frame(runs, PSI_COLS)
    quality = bi_export.quality_long(runs, PSI_COLS)
    nulls = quality[quality["metric"] == "null_rate"]
    for position, (_, run) in enumerate(runs.iterrows()):
        for column in PSI_COLS:
            expected = wide.iloc[position][column]
            match = nulls[
                (nulls["run_id"] == run["run_id"]) & (nulls["column"] == column)
            ]
            assert (
                match.empty if pd.isna(expected) else match["value"].iloc[0] == expected
            )


def test_alerts_long_is_one_row_per_alert_and_empty_when_none():
    alerts = bi_export.alerts_long(_runs())
    assert list(alerts.columns) == list(bi_export.ALERT_COLUMNS)
    assert alerts["alert"].tolist() == [
        "PSI drift on price: 0.350 > 0.2",
        "null-rate drift on vin: +0.120",
    ]
    quiet = _runs().drop(index=3).reset_index(drop=True)  # drop the only alerting run
    assert bi_export.alerts_long(quiet).empty


def test_null_rate_delta_frame_unions_columns_across_runs():
    """null_rate_delta spans every shared column, not a configured shortlist, so it varies per run."""
    runs = _runs()
    wide = frames.null_rate_delta_frame(runs, id_columns=bi_export.ID_COLUMNS)
    assert list(wide.columns) == ["run_id", "batch_label", "mileage", "price", "vin"]
    long = bi_export.drift_long(runs, _drift_cfg())
    deltas = long[long["metric"] == "null_rate_delta"]
    # W03 recorded mileage+price, W04 price+vin, the re-run price only -- absences are dropped
    assert deltas.groupby("run_id").size().to_dict() == {
        "2024-W03-computed": 2,
        "2024-W04-computed": 2,
        "2024-W04-rerun": 1,
    }


# ---------------------------------------------------------------- model comparison (other grain)


def test_model_comparison_reads_metrics_json_and_flags_the_winner():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.json"
        path.write_text(
            json.dumps(
                {
                    "models": {
                        "mean_baseline": {"mae": 10207.0, "rmse": 14174.0},
                        "random_forest": {"mae": 5833.0, "rmse": 9736.0},
                    },
                    "winner": "random_forest",
                    "beats_baseline": True,
                    "baseline_mae": 10207.0,
                }
            )
        )
        table = bi_export.model_comparison(path)
    assert list(table.columns) == list(bi_export.MODEL_COLUMNS)
    assert len(table) == 4  # 2 models x 2 metrics
    assert set(table[table["is_winner"]]["model"]) == {"random_forest"}
    assert table["beats_baseline"].all()
    mae = table[table["metric"] == "mae"].set_index("model")["value"]
    assert mae["random_forest"] == 5833.0 and mae["mean_baseline"] == 10207.0


def test_model_comparison_is_header_only_when_metrics_json_is_absent():
    table = bi_export.model_comparison(Path("does-not-exist.json"))
    assert table.empty
    assert list(table.columns) == list(bi_export.MODEL_COLUMNS)


# ----------------------------------------------------------------------- the data dictionary


def test_data_dictionary_covers_every_exported_column_exactly_once():
    """Generated, not hand-maintained: it cannot silently fall behind a schema change."""
    tables = bi_export.build_tables(_runs(), load_settings(), "runs")
    documented = bi_export.data_dictionary()
    assert not documented.duplicated(subset=["table", "column"]).any()
    assert set(documented["table"]) == set(tables)
    for name, frame in tables.items():
        rows = documented[documented["table"] == name]
        assert list(rows["column"]) == list(frame.columns), name


def test_data_dictionary_records_a_denominator_for_every_rate_column():
    documented = bi_export.data_dictionary()
    rates = documented[documented["column"].str.endswith("_rate")]
    assert len(rates) == 3  # quarantine_rate, row_pass_rate, dq_check_pass_rate
    assert (rates["denominator"].str.len() > 0).all()
    check_level = rates[rates["column"] == "dq_check_pass_rate"].iloc[0]
    assert "check" in check_level["denominator"]
    assert (
        "n_rows" in rates[rates["column"] == "quarantine_rate"].iloc[0]["denominator"]
    )


# ------------------------------------------------------------------------ the end-to-end export


def _seed(engine: Engine, table: str) -> None:
    """Persist two real RunRecords -- a baseline and a drifting one -- through save_run."""
    baseline = RunRecord.new("2024-W01")
    baseline.n_rows, baseline.n_quarantined = 500, 50
    baseline.dq_pass_rate, baseline.mae = 0.75, 900.0
    baseline.col_stats = _stats(0.02)
    save_run(baseline, engine, table)

    drifting = RunRecord.new("2024-W02")
    drifting.n_rows, drifting.n_quarantined = 600, 60
    drifting.dq_pass_rate, drifting.mae = 0.875, 950.0
    drifting.col_stats = _stats(0.04)
    drifting.drift = {"psi": {"price": 0.35}, "alerts": ["PSI drift on price"]}
    save_run(drifting, engine, table)


def test_export_writes_every_configured_table_into_bi_dir():
    with _engine() as engine, tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "bi"
        settings = _settings_into(out)
        _seed(engine, settings["database"]["runs_table"])
        written = bi_export.export_bi_tables(engine, settings)
    assert set(written) == set(settings["reporting"]["tables"])
    assert set(written) == set(bi_export.build_tables(pd.DataFrame(), settings, "runs"))


def test_export_creates_the_output_directory_and_writes_readable_csvs():
    with _engine() as engine, tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "nested" / "bi"  # does not exist yet
        settings = _settings_into(out)
        _seed(engine, settings["database"]["runs_table"])
        written = bi_export.export_bi_tables(engine, settings)
        assert out.is_dir()
        runs = pd.read_csv(written["runs"])
        alerts = pd.read_csv(written["drift_alerts"])
        for path in written.values():
            assert path.is_file() and pd.read_csv(path) is not None
    assert len(runs) == 2
    assert runs["drift_status"].tolist() == ["baseline", "computed"]
    assert runs["quarantine_rate"].tolist() == [0.1, 0.1]
    assert alerts["alert"].tolist() == ["PSI drift on price"]


def test_export_is_valid_when_there_is_no_run_history():
    """An empty history exports header-only files, so a BI refresh degrades rather than breaking."""
    with _engine() as engine, tempfile.TemporaryDirectory() as directory:
        settings = _settings_into(Path(directory) / "bi")
        written = bi_export.export_bi_tables(engine, settings)  # no runs table at all
        runs = pd.read_csv(written["runs"])
        manifest = pd.read_csv(written["export_manifest"])
        dictionary = pd.read_csv(written["data_dictionary"])
    assert runs.empty and list(runs.columns) == list(bi_export.RUNS_COLUMNS)
    assert manifest["n_runs"].tolist() == [0]
    assert not dictionary.empty  # the dictionary is static and always exports


def test_cli_returns_zero_and_writes_tables(monkeypatch, capsys):
    with _engine() as engine, tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "bi"
        settings = _settings_into(out)
        _seed(engine, settings["database"]["runs_table"])
        monkeypatch.setattr(bi_export, "load_settings", lambda: settings)
        monkeypatch.setattr(bi_export.database, "get_engine", lambda: engine)
        assert bi_export._cli() == 0
        assert (out / "runs.csv").is_file()
    assert "BI export: 7 table(s)" in capsys.readouterr().out


@pytest.mark.parametrize("table", LONG_TABLES)
def test_every_long_table_relates_to_a_real_run_or_stands_alone(table):
    """Fact tables key on run_id; model_comparison deliberately does not (a different grain)."""
    tables = bi_export.build_tables(_runs(), load_settings(), "runs")
    frame = tables[table]
    if table == "model_comparison":
        assert "run_id" not in frame.columns
    else:
        assert set(frame["run_id"]) <= set(_runs()["run_id"])
