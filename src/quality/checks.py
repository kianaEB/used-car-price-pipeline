"""Data-quality checks -- the heart of the pipeline.

Each check is a small, independent function that takes a DataFrame (+ params from config) and
returns a `CheckResult`. Checks never mutate the data; they only *report*. `report.py` runs them
and decides whether to halt the pipeline (see SPEC 7).

Two checks below (`check_schema`, `check_min_rows`) are fully implemented to establish the
pattern; the rest are stubs with a clear contract. Every check must have a matching test in
`tests/test_quality_checks.py` proving it passes on clean data and fails on the right defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Severity = Literal["ERROR", "WARN", "INFO"]


@dataclass
class CheckResult:
    """Outcome of a single data-quality check."""

    name: str
    passed: bool
    severity: Severity
    n_violations: int
    n_rows: int
    detail: str = ""
    sample: list[dict] = field(default_factory=list)


def _result(
    name: str,
    passed: bool,
    severity: Severity,
    n_violations: int,
    df: pd.DataFrame,
    detail: str = "",
    offenders: pd.DataFrame | None = None,
) -> CheckResult:
    """Build a CheckResult, attaching up to 5 offending rows as a debugging sample."""
    sample: list[dict] = []
    if offenders is not None and len(offenders) > 0:
        sample = offenders.head(5).to_dict(orient="records")
    return CheckResult(name, passed, severity, int(n_violations), len(df), detail, sample)


def check_schema(df: pd.DataFrame, required_columns: list[str]) -> CheckResult:
    """ERROR if any required column is missing. TODO: also verify dtype coercibility."""
    missing = [c for c in required_columns if c not in df.columns]
    detail = f"missing columns: {missing}" if missing else "all required columns present"
    return _result("schema", not missing, "ERROR", len(missing), df, detail=detail)


def check_min_rows(df: pd.DataFrame, min_rows: int) -> CheckResult:
    """ERROR if the dataset has fewer than `min_rows` rows (truncated/empty ingest)."""
    ok = len(df) >= min_rows
    return _result("min_rows", ok, "ERROR", 0 if ok else 1, df, detail=f"{len(df)} rows (min {min_rows})")


def check_nulls(df: pd.DataFrame, null_fraction_max: dict[str, float]) -> CheckResult:
    """ERROR/WARN if a column's null fraction exceeds its configured threshold.

    TODO: for each column in `null_fraction_max`, compute null fraction; collect violations;
    severity ERROR for price/year, WARN otherwise (or drive severity from config).
    """
    raise NotImplementedError


def check_ranges(df: pd.DataFrame, ranges: dict[str, list]) -> CheckResult:
    """ERROR if price / year / mileage fall outside their [min, max] domain bounds.

    This is the check that catches the prototype's negative/garbage values.
    TODO: count rows violating any configured range; sample offenders.
    """
    raise NotImplementedError


def check_duplicates(df: pd.DataFrame, vin_severity: Severity = "ERROR") -> CheckResult:
    """Flag fully duplicated rows and (if `vin` present) repeated VINs. TODO: implement."""
    raise NotImplementedError


def check_categories(df: pd.DataFrame, column: str, known: list[str]) -> CheckResult:
    """WARN if `column` contains values outside the `known` set. Skip if column absent. TODO."""
    raise NotImplementedError


def check_consistency(df: pd.DataFrame) -> CheckResult:
    """WARN on cross-field contradictions, e.g. 'new' listings with non-zero mileage.

    Mirrors the exact bug class in the prototype (mileage handling). TODO: implement;
    skip gracefully if the required columns are absent.
    """
    raise NotImplementedError
