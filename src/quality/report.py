"""Two-layer data-quality validation: run it, print a summary, save JSON, and enforce the gate.

Layer 1 = pandera schema (quality/schema.py, technical). Layer 2 = business rules (quality/checks.py).
CLI:  python -m src.quality.report      # validates the configured raw file; exit 1 on ERROR

The exit code is non-zero if any ERROR-severity check fails, so `make validate`, CI, and the
pipeline can all gate on data quality. `run_report` is what the pipeline calls directly.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config import Settings, load_settings
from src.ingest.dataset import load_dataset
from src.quality import checks, schema

log = logging.getLogger("quality.report")


@dataclass
class DataQualityReport:
    """Aggregated result of a validation run (Layer 1 schema + Layer 2 business rules)."""

    results: list[checks.CheckResult] = field(default_factory=list)

    @property
    def errors(self) -> list[checks.CheckResult]:
        """ERROR-severity checks that failed."""
        return [r for r in self.results if r.severity == "ERROR" and not r.passed]

    @property
    def passed(self) -> bool:
        """True if no ERROR-severity check failed (WARN failures are allowed through)."""
        return len(self.errors) == 0

    def to_json(self, path: str | Path) -> None:
        """Persist the full result list to JSON (for inspection / the README catch-rate)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([asdict(r) for r in self.results], indent=2))

    def print_summary(self) -> None:
        """Print a one-line-per-check summary table."""
        for r in self.results:
            status = "PASS" if r.passed else r.severity
            print(f"[{status:5}] {r.name:14} {r.detail}")


def run_report(df: pd.DataFrame, settings: Settings | None = None) -> DataQualityReport:
    """Run Layer 1 (pandera schema) then Layer 2 (business rules); return a DataQualityReport.

    Layer 1 is the holistic technical gate; Layer 2 adds itemised, actionable rules (per-field
    ranges, per-column null policy, duplicates, category membership, cross-field consistency).
    Some Layer-2 rules deliberately overlap Layer 1 -- defence in depth plus readable detail.
    """
    settings = settings if settings is not None else load_settings()
    quality = settings["quality"]
    ranges = quality["ranges"]
    consistency = quality.get("consistency", {})
    results = [
        checks.check_schema(
            df, settings["schema"]["required_columns"]
        ),  # columns present
        schema.validate_schema(df, ranges),  # Layer 1: pandera technical schema
        checks.check_min_rows(df, quality["min_rows"]),
        checks.check_nulls(df, quality["null_fraction_max"]),
        checks.check_ranges(df, ranges),
        checks.check_duplicates(df, quality.get("duplicate_vin_severity", "ERROR")),
        checks.check_categories(df, "title_status", quality["known_title_status"]),
        checks.check_consistency(
            df,
            consistency.get("fresh_conditions", []),
            consistency.get("max_fresh_mileage", float("inf")),
        ),
    ]
    report = DataQualityReport(results=results)
    n_passed = sum(r.passed for r in results)
    n_warn = sum(1 for r in results if r.severity == "WARN" and not r.passed)
    log.info(
        "DQ report: %d/%d checks passed, %d error(s), %d warn(s)",
        n_passed,
        len(results),
        len(report.errors),
        n_warn,
    )
    return report


def main() -> int:
    """CLI entry: load the raw file, validate, save the report, and gate on ERROR severity."""
    settings = load_settings()
    df = load_dataset(settings["paths"]["raw_file"])
    report = run_report(df, settings)
    report.print_summary()
    report.to_json(settings["paths"]["dq_report"])
    if not report.passed:
        print("DATA QUALITY GATE FAILED -- halting before DB load / training.")
        return 1
    print("Data quality OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
