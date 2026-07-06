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


def compute_drift(previous: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> DriftReport:
    """Compare previous vs current batch: PSI (numeric), null-rate, category shift, freshness.

    TODO: for each column in cfg['psi_columns'] compute psi() and alert if > cfg['psi_alert'];
    compute per-column null-rate deltas (alert > cfg['null_rate_delta_alert']); top-category share
    shift for brand/title_status (alert > cfg['category_shift_alert']); assemble DriftReport.alerts.
    """
    raise NotImplementedError
