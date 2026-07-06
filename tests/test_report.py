"""Tests for run_report + the DQ gate: ERROR halts, WARN-only passes, and defects are caught exactly.

The gate is the whole point of the DQ layer -- these prove an ERROR-severity failure flips
report.passed to False (the pipeline halts) while a WARN-only failure leaves it True. The
catch-exactly test uses the generator's df.attrs ground truth to show each injected defect maps to
a specific check's n_violations. Uses stdlib tempfile (the machine's pytest tmp_path base is locked).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.config import load_settings
from src.ingest.dataset import generate_synthetic
from src.quality import checks
from src.quality.report import run_report
from src.quality.schema import validate_schema

EXPECTED_CHECKS = {
    "schema",
    "min_rows",
    "nulls",
    "ranges",
    "duplicates",
    "category:title_status",
    "consistency",
}


def test_run_report_runs_both_layers_and_passes_on_clean() -> None:
    """A clean batch passes the gate; every Layer-1 + Layer-2 check is present in the report."""
    df = generate_synthetic(n=600, seed=42, bad_fraction=0.0)
    report = run_report(df, load_settings())
    assert report.passed
    assert not report.errors
    assert EXPECTED_CHECKS <= {r.name for r in report.results}


def test_gate_halts_on_error_severity_failure() -> None:
    """Injected ERROR-class defects flip report.passed to False -- the pipeline's hard stop."""
    df = generate_synthetic(n=4000, seed=42, bad_fraction=0.06)
    report = run_report(df, load_settings())
    assert not report.passed
    error_names = {e.name for e in report.errors}
    assert {"schema", "ranges", "duplicates"} <= error_names


def test_gate_allows_warn_only_batch() -> None:
    """A WARN-only defect (unknown title_status) is logged but does NOT halt the pipeline."""
    df = generate_synthetic(n=600, seed=42, bad_fraction=0.0)
    df.loc[:20, "title_status"] = "flooded"  # outside the known set -> category WARN
    report = run_report(df, load_settings())
    assert report.passed  # no ERROR -> gate stays open
    assert not report.errors
    warned = [r for r in report.results if not r.passed and r.severity == "WARN"]
    assert any(r.name == "category:title_status" for r in warned)


def test_layer_catches_exactly_the_injected_defects() -> None:
    """Each injected defect (df.attrs ground truth) maps to a specific check's n_violations."""
    df = generate_synthetic(n=4000, seed=42, bad_fraction=0.06)
    defects = df.attrs["defects"]
    ranges = load_settings()["quality"]["ranges"]

    assert checks.check_ranges(df, ranges).n_violations == (
        defects["nonpositive_price"] + defects["impossible_year"]
    )
    assert checks.check_duplicates(df).n_violations == defects["duplicate_vin"]
    assert checks.check_nulls(df, {"brand": 0.0}).n_violations == defects["null_brand"]

    schema_result = validate_schema(df, ranges)
    assert not schema_result.passed
    assert schema_result.n_violations >= (
        defects["null_brand"]
        + defects["nonpositive_price"]
        + defects["impossible_year"]
    )


def test_print_summary_emits_a_line_per_check(capsys) -> None:
    """The human-readable summary prints one status line per check."""
    df = generate_synthetic(n=200, seed=42, bad_fraction=0.06)
    report = run_report(df, load_settings())
    report.print_summary()
    out = capsys.readouterr().out
    assert out.count("\n") >= len(report.results)


def test_report_to_json_is_valid_and_reloadable() -> None:
    """to_json writes a valid JSON array of check dicts (offender samples included) to disk."""
    df = generate_synthetic(n=500, seed=42, bad_fraction=0.06)
    report = run_report(df, load_settings())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dq_report.json"
        report.to_json(path)
        loaded = json.loads(path.read_text())
    assert isinstance(loaded, list) and len(loaded) == len(report.results)
    assert {"name", "passed", "severity", "n_violations", "sample"} <= set(loaded[0])
