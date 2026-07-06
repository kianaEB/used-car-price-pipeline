"""Drift & freshness monitoring: compare the current batch to the previous run.

`psi()` (Population Stability Index) is fully implemented and unit-tested: ~0 for identical
distributions, large for a known shift. The higher-level `compute_drift` orchestration is a stub
with a clear contract. Drift is a *signal*, not a hard gate (the DQ ERROR gate is the hard stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    """Per-signal drift result plus an overall alert flag."""

    psi: dict[str, float] = field(default_factory=dict)
    null_rate_delta: dict[str, float] = field(default_factory=dict)
    category_shift: dict[str, float] = field(default_factory=dict)
    freshness_days: float = 0.0
    alerts: list[str] = field(default_factory=list)

    @property
    def has_alert(self) -> bool:
        """True if any monitored signal breached its configured threshold."""
        return len(self.alerts) > 0


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions.

    ~0 means no shift; > 0.2 conventionally signals meaningful drift. Bin edges are quantiles of
    `expected`; a small epsilon avoids division-by-zero on empty bins. NaNs are dropped.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    eps = 1e-6
    e_pct = e_counts / max(e_counts.sum(), 1) + eps
    a_pct = a_counts / max(a_counts.sum(), 1) + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def null_rate_delta(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Per-shared-column change in null fraction (current minus previous); signed."""
    deltas: dict[str, float] = {}
    for col in current.columns:
        if col not in previous.columns:
            continue
        prev_rate = float(previous[col].isna().mean()) if len(previous) else 0.0
        curr_rate = float(current[col].isna().mean()) if len(current) else 0.0
        deltas[col] = curr_rate - prev_rate
    return deltas


def category_shift(
    previous: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    """Largest change in any single category's share per column (catches a new category too)."""
    shifts: dict[str, float] = {}
    for col in columns:
        if col not in previous.columns or col not in current.columns:
            continue
        prev_share = previous[col].value_counts(normalize=True)
        curr_share = current[col].value_counts(normalize=True)
        categories = set(prev_share.index) | set(curr_share.index)
        shifts[col] = max(
            (abs(curr_share.get(c, 0.0) - prev_share.get(c, 0.0)) for c in categories),
            default=0.0,
        )
    return shifts


def freshness(
    current: pd.DataFrame, reference_date: str | None, date_column: str = "posting_date"
) -> float:
    """Age in days of the newest posting_date versus the config reference date (not wall-clock now)."""
    if reference_date is None or date_column not in current.columns:
        return 0.0
    newest = pd.to_datetime(current[date_column], errors="coerce").max()
    if pd.isna(newest):
        return 0.0
    return float((pd.Timestamp(reference_date) - newest).days)


def compute_drift(
    previous: pd.DataFrame, current: pd.DataFrame, cfg: dict
) -> DriftReport:
    """Compare current batch to the previous run across PSI, null-rate, category shift, freshness.

    `cfg` is the `monitoring` config block (drift thresholds + freshness). Drift is a SIGNAL, not a
    gate: this only records values and appends alert strings -- it never raises or halts the
    pipeline (the DQ ERROR gate is the only hard stop).
    """
    drift_cfg = cfg.get("drift", {})
    fresh_cfg = cfg.get("freshness", {})
    bins = int(drift_cfg.get("psi_bins", 10))
    psi_alert = float(drift_cfg.get("psi_alert", 0.2))
    null_alert = float(drift_cfg.get("null_rate_delta_alert", 0.10))
    category_alert = float(drift_cfg.get("category_shift_alert", 0.15))
    max_staleness = float(fresh_cfg.get("max_staleness_days", 30))
    reference_date = fresh_cfg.get("reference_date")

    report = DriftReport()

    for col in drift_cfg.get("psi_columns", []):
        if col in previous.columns and col in current.columns:
            value = psi(previous[col].to_numpy(), current[col].to_numpy(), bins)
            report.psi[col] = value
            if value > psi_alert:
                report.alerts.append(f"PSI drift on {col}: {value:.3f} > {psi_alert}")

    report.null_rate_delta = null_rate_delta(previous, current)
    for col, delta in report.null_rate_delta.items():
        if abs(delta) > null_alert:
            report.alerts.append(f"null-rate drift on {col}: {delta:+.3f}")

    report.category_shift = category_shift(
        previous, current, drift_cfg.get("category_columns", [])
    )
    for col, shift in report.category_shift.items():
        if shift > category_alert:
            report.alerts.append(
                f"category shift on {col}: {shift:.3f} > {category_alert}"
            )

    report.freshness_days = freshness(current, reference_date)
    if reference_date is not None and report.freshness_days > max_staleness:
        report.alerts.append(
            f"stale data: newest posting_date {report.freshness_days:.0f}d old "
            f"> {max_staleness:.0f}d"
        )

    return report
