"""Two-layer data-quality validation: run it, print a summary, save JSON, and enforce the gate.

Layer 1 = pandera schema (quality/schema.py, technical). Layer 2 = business rules (quality/checks.py).
CLI:  python -m src.quality.report      # validates the configured raw file; exit 1 on ERROR

The exit code is non-zero if any ERROR-severity check fails, so `make validate`, CI, and the
pipeline can all gate on data quality. `run_report` is what the pipeline calls directly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config import Settings, load_settings
from src.ingest.dataset import load_dataset
from src.quality import checks


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

    TODO:
      - Layer 1: schema.validate_schema(df, settings['quality']['ranges']) -> one CheckResult
      - Layer 2 (from checks.py): check_min_rows, check_nulls, check_duplicates,
        check_categories(title_status, known set), check_consistency -> CheckResults
      - collect everything into DataQualityReport(results=[...]) in a stable order
    """
    raise NotImplementedError


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
