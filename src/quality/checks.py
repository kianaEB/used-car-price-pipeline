"""Data-quality checks -- the heart of the pipeline.

Each check is a small, independent function that takes a DataFrame (+ params from config) and
returns a `CheckResult`. Checks never mutate the data; they only *report*. `report.py` runs them
and decides whether to halt the pipeline (see SPEC 7).

Layer 1 (technical) lives in schema.py (pandera). These are the Layer-2 *business* rules: shape
(min rows), per-column null policy, domain ranges, duplicate rows/VINs, category membership, and
cross-field consistency -- the things a declarative schema can't cleanly express.
"""

from __future__ import annotations

import json
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
    offending_index: list[int] = field(default_factory=list)


def _sample(offenders: pd.DataFrame | None) -> list[dict]:
    """Up to 5 offending rows as JSON-safe dicts (Timestamps->ISO, NaN->None) for the report."""
    if offenders is None or len(offenders) == 0:
        return []
    return json.loads(offenders.head(5).to_json(orient="records", date_format="iso"))


def _offending(offenders: pd.DataFrame | None) -> list[int]:
    """Row-index labels of ALL offending rows -- the single source of truth for quarantine."""
    if offenders is None or len(offenders) == 0:
        return []
    return [int(i) for i in offenders.index.tolist()]


def _result(
    name: str,
    passed: bool,
    severity: Severity,
    n_violations: int,
    df: pd.DataFrame,
    detail: str = "",
    offenders: pd.DataFrame | None = None,
) -> CheckResult:
    """Build a CheckResult with up to 5 sample rows and the full offending-row index set."""
    return CheckResult(
        name,
        passed,
        severity,
        int(n_violations),
        len(df),
        detail,
        _sample(offenders),
        _offending(offenders),
    )


def check_schema(df: pd.DataFrame, required_columns: list[str]) -> CheckResult:
    """ERROR if any required column is missing (structural precondition for every other check)."""
    missing = [c for c in required_columns if c not in df.columns]
    detail = (
        f"missing columns: {missing}" if missing else "all required columns present"
    )
    return _result("columns", not missing, "ERROR", len(missing), df, detail=detail)


def check_min_rows(df: pd.DataFrame, min_rows: int) -> CheckResult:
    """ERROR if the dataset has fewer than `min_rows` rows (truncated/empty ingest)."""
    ok = len(df) >= min_rows
    return _result(
        "min_rows",
        ok,
        "ERROR",
        0 if ok else 1,
        df,
        detail=f"{len(df)} rows (min {min_rows})",
    )


def check_nulls(df: pd.DataFrame, null_fraction_max: dict[str, float]) -> CheckResult:
    """ERROR/WARN if a column's null fraction exceeds its configured threshold.

    A zero-tolerance column (threshold 0.0, e.g. price) escalates the check to ERROR; columns with
    a non-zero tolerance only WARN when breached. n_violations counts the offending null cells.
    """
    n_rows = len(df)
    violated: dict[str, tuple[float, int]] = {}
    is_error = False
    for col, threshold in null_fraction_max.items():
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        fraction = n_null / n_rows if n_rows else 0.0
        if fraction > threshold:
            violated[col] = (fraction, n_null)
            if threshold == 0.0:
                is_error = True
    if not violated:
        return _result(
            "nulls", True, "WARN", 0, df, detail="null fractions within thresholds"
        )
    mask = pd.Series(False, index=df.index)
    for col in violated:
        mask |= df[col].isna()
    detail = "; ".join(
        f"{col}: {frac:.1%} > {null_fraction_max[col]:.0%}"
        for col, (frac, _) in violated.items()
    )
    n_violations = sum(n_null for _, n_null in violated.values())
    severity: Severity = "ERROR" if is_error else "WARN"
    return _result("nulls", False, severity, n_violations, df, detail, df.loc[mask])


def check_ranges(df: pd.DataFrame, ranges: dict[str, list]) -> CheckResult:
    """ERROR if price / year / mileage fall outside their configured [min, max] domain bounds.

    This is the check that catches the prototype's negative/garbage values. NaNs are ignored here
    (nullability is the null check's / schema's job), so a row is counted once if any field is out
    of range.
    """
    mask = pd.Series(False, index=df.index)
    parts: list[str] = []
    for col, bounds in ranges.items():
        if col not in df.columns:
            continue
        low, high = bounds
        values = pd.to_numeric(df[col], errors="coerce")
        col_mask = ((values < low) | (values > high)).fillna(False)
        n_col = int(col_mask.sum())
        if n_col:
            parts.append(f"{col}: {n_col} outside [{low}, {high}]")
        mask |= col_mask
    n_violations = int(mask.sum())
    detail = "; ".join(parts) if parts else "all values within range"
    return _result(
        "ranges", n_violations == 0, "ERROR", n_violations, df, detail, df.loc[mask]
    )


def check_duplicates(df: pd.DataFrame, vin_severity: Severity = "ERROR") -> CheckResult:
    """Flag fully duplicated rows and (if `vin` present) repeated VINs; every offender is counted."""
    mask = df.duplicated(keep=False)
    parts: list[str] = []
    n_full = int(mask.sum())
    if n_full:
        parts.append(f"{n_full} fully-duplicated row(s)")
    if "vin" in df.columns:
        vin_mask = df["vin"].notna() & df["vin"].duplicated(keep=False)
        n_vin = int(vin_mask.sum())
        if n_vin:
            parts.append(f"{n_vin} row(s) with a repeated VIN")
        mask = mask | vin_mask
    n_violations = int(mask.sum())
    detail = "; ".join(parts) if parts else "no duplicate rows or VINs"
    return _result(
        "duplicates",
        n_violations == 0,
        vin_severity,
        n_violations,
        df,
        detail,
        df.loc[mask],
    )


def check_categories(df: pd.DataFrame, column: str, known: list[str]) -> CheckResult:
    """WARN if `column` holds values outside the `known` set. Passes (skipped) if column absent."""
    if column not in df.columns:
        return _result(
            f"category:{column}",
            True,
            "WARN",
            0,
            df,
            detail=f"{column} absent; skipped",
        )
    values = df[column]
    mask = values.notna() & ~values.isin(set(known))
    n_violations = int(mask.sum())
    detail = (
        f"{n_violations} value(s) outside {known}"
        if n_violations
        else f"all {column} values known"
    )
    return _result(
        f"category:{column}",
        n_violations == 0,
        "WARN",
        n_violations,
        df,
        detail,
        df.loc[mask],
    )


def check_consistency(
    df: pd.DataFrame, fresh_conditions: list[str], max_mileage: float
) -> CheckResult:
    """WARN on cross-field contradictions: a 'fresh' condition paired with very high mileage.

    Mirrors the prototype's mileage-handling bug class. Passes (skipped) if condition/mileage absent.
    """
    if "condition" not in df.columns or "mileage" not in df.columns:
        return _result(
            "consistency",
            True,
            "WARN",
            0,
            df,
            detail="condition/mileage absent; skipped",
        )
    fresh = {c.lower() for c in fresh_conditions}
    is_fresh = df["condition"].astype("string").str.lower().isin(fresh)
    mileage = pd.to_numeric(df["mileage"], errors="coerce")
    mask = (is_fresh & (mileage > max_mileage)).fillna(False)
    n_violations = int(mask.sum())
    detail = (
        f"{n_violations} '{'/'.join(fresh_conditions)}' listing(s) over {max_mileage:g} miles"
        if n_violations
        else "condition and mileage consistent"
    )
    return _result(
        "consistency", n_violations == 0, "WARN", n_violations, df, detail, df.loc[mask]
    )
