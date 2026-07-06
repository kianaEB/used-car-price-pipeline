"""Layer 1 of data quality: a declarative pandera schema (the 'technical' checks).

Encodes columns, dtypes, nullability, and numeric ranges. Validate with lazy=True so ALL failures
are collected at once, then folded into a single CheckResult for the unified report. The 'business'
rules (consistency, category membership, cross-run drift) live in checks.py / monitoring.
"""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.quality.checks import CheckResult


def build_schema(ranges: dict[str, list], current_year: int = 2027) -> DataFrameSchema:
    """Return the technical schema for the canonical used-car columns."""
    p_lo, p_hi = ranges.get("price", [1, 1_000_000])
    y_lo, y_hi = ranges.get("year", [1950, current_year])
    m_lo, m_hi = ranges.get("mileage", [0, 1_000_000])
    return DataFrameSchema(
        {
            "price": Column(
                float, Check.in_range(p_lo, p_hi), nullable=False, coerce=True
            ),
            "brand": Column(str, nullable=False, coerce=True),
            "model": Column(str, nullable=True, coerce=True),
            "year": Column(
                int, Check.in_range(y_lo, y_hi), nullable=False, coerce=True
            ),
            "mileage": Column(
                float, Check.in_range(m_lo, m_hi), nullable=True, coerce=True
            ),
        },
        strict=False,  # allow extra columns (vin, state, posting_date, ...)
        coerce=True,
    )


def validate_schema(
    df, ranges: dict[str, list], current_year: int = 2027
) -> CheckResult:
    """Run the pandera schema (lazy) and fold all failures into one technical CheckResult."""
    schema = build_schema(ranges, current_year)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases
        n = int(len(fc))
        sample = [
            {k: str(v) for k, v in row.items()} for row in fc.head(5).to_dict("records")
        ]
        return CheckResult(
            "schema", False, "ERROR", n, len(df), f"{n} schema violation(s)", sample
        )
    return CheckResult("schema", True, "ERROR", 0, len(df), "pandera schema OK")
