"""Load the used-car dataset for the pipeline.

DEFAULT source is 'synthetic' (config: dataset.source) -- the pipeline runs end to end with NO
download and NO accounts. `load(settings)` dispatches on the configured source. A real dataset is
optional (see src/ingest/download.py and `make data-real`).

CLI:
    python -m src.ingest.dataset --synthetic --out data/raw/sample.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings

CANONICAL_COLUMNS = [
    "price",
    "brand",
    "model",
    "year",
    "mileage",
    "title_status",
    "vin",
    "posting_date",
]

# Source-column -> canonical name used when a caller passes no explicit column_map (mirrors the
# `column_map` block in config.yaml). Keeps report.py's single-arg load_dataset() call working.
DEFAULT_COLUMN_MAP = {"manufacturer": "brand", "odometer": "mileage"}

# --- Synthetic-fixture constants (the SHAPE of the fake data, not tunable pipeline params). The
# tunable knobs -- n, seed, bad_fraction, n_weeks, drift -- all come from config via load(). ---
_SYNTHETIC_START = pd.Timestamp(
    "2024-01-01"
)  # week-0 anchor; fixed so runs are reproducible
_STALE_OFFSET = pd.Timedelta(
    days=5 * 365
)  # how far back a "stale posting_date" defect is pushed
_DUPLICATE_VIN = "DUPLICATE-VIN"  # sentinel VIN shared by the duplicate-VIN defect rows
_IMPOSSIBLE_YEARS = [
    1400,
    1800,
    1900,
    2035,
    2100,
    3000,
]  # all outside any sane year range

_BRAND_MODELS: dict[str, list[str]] = {
    "toyota": ["corolla", "camry", "rav4"],
    "ford": ["focus", "f-150", "escape"],
    "honda": ["civic", "accord", "cr-v"],
    "chevrolet": ["malibu", "silverado", "equinox"],
    "bmw": ["3 series", "5 series", "x5"],
    "nissan": ["altima", "sentra", "rogue"],
    "volkswagen": ["jetta", "passat", "tiguan"],
    "hyundai": ["elantra", "sonata", "tucson"],
}
_NEW_BRAND = (
    "rivian"  # appears mid-stream (drift.new_brand_week) to create a category shift
)
_NEW_BRAND_MODELS = ["r1t", "r1s"]
_BRAND_PREMIUM: dict[str, float] = {
    "toyota": 3000.0,
    "ford": 2500.0,
    "honda": 2800.0,
    "chevrolet": 2200.0,
    "bmw": 12000.0,
    "nissan": 2000.0,
    "volkswagen": 3500.0,
    "hyundai": 1800.0,
    _NEW_BRAND: 25000.0,
}
_TITLE_STATUSES = ["clean", "salvage", "rebuilt", "lien", "missing", "parts only"]
_TITLE_WEIGHTS = [0.85, 0.05, 0.04, 0.03, 0.02, 0.01]
_CONDITIONS = ["new", "like new", "excellent", "good", "fair", "salvage"]
_CONDITION_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]
_STATES = ["ca", "tx", "ny", "fl", "wa", "il", "pa", "oh"]


def load(settings: Settings) -> pd.DataFrame:
    """Return the working DataFrame per settings['dataset']['source'].

    - 'synthetic' (default): generate in-memory; no I/O, no network, KNOWN ground truth.
    - 'url' | 'kaggle' | 'openml': read the file cached at paths.raw_file (fetch it first with
      `make data-real`, i.e. src.ingest.download).
    """
    ds = settings["dataset"]
    if ds.get("source", "synthetic") == "synthetic":
        return generate_synthetic(
            n=ds.get("n_rows", 8000),
            seed=settings.seed,
            bad_fraction=ds.get("bad_fraction", 0.06),
            n_weeks=ds.get("n_weeks", 8),
            drift=ds.get("drift", {}),
        )
    return load_dataset(settings["paths"]["raw_file"], settings["column_map"])


def load_dataset(
    path: str | Path, column_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Read a real CSV and normalise columns to the canonical schema without dropping bad rows."""
    mapping = DEFAULT_COLUMN_MAP if column_map is None else column_map
    df = pd.read_csv(Path(path))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={str(k).lower(): v for k, v in mapping.items()})
    # Coerce numerics but keep offending values as NaN -- the DQ layer reports them, it doesn't drop.
    for col in ("price", "mileage", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "posting_date" in df.columns:
        # Real feeds mix timezone offsets per row; normalise to UTC then drop tz so the rest of the
        # pipeline (batching, freshness vs a naive reference date) stays tz-consistent with synthetic.
        df["posting_date"] = pd.to_datetime(
            df["posting_date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    return df


def generate_synthetic(
    n: int = 8000,
    seed: int = 42,
    bad_fraction: float = 0.06,
    n_weeks: int = 8,
    drift: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Return a realistic, MESSY synthetic used-car dataset with KNOWN defect/drift ground truth.

    Why synthetic-by-default: it makes the whole project self-contained (no download / account),
    reproducible, and -- crucially for a data-QUALITY project -- gives KNOWN defect and drift
    ground truth, so the DQ catch-rate and drift alerts are exactly measurable. Injected-defect
    counts are recorded on ``df.attrs`` so the README catch-rate can be exact.
    """
    rng = np.random.default_rng(int(seed))
    drift = dict(drift or {})
    n = int(n)
    n_weeks = max(int(n_weeks), 1)
    inflation = float(drift.get("price_inflation_per_week", 0.0))
    rising_null_col = drift.get("rising_null_column")
    new_brand_week = drift.get("new_brand_week")
    shock_week = drift.get("price_shock_week")
    shock_multiplier = float(drift.get("price_shock_multiplier", 1.0))
    reference_year = _SYNTHETIC_START.year

    week = rng.integers(0, n_weeks, size=n)
    day_offset = rng.integers(0, 7, size=n)
    posting_date = _SYNTHETIC_START + pd.to_timedelta(week * 7 + day_offset, unit="D")

    year = rng.integers(1995, reference_year + 1, size=n)
    age = reference_year - year
    mileage = np.clip(age * 11000 + rng.normal(0, 12000, size=n), 0, None)

    brand_names = list(_BRAND_MODELS)
    brand = rng.choice(brand_names, size=n).astype(object)
    if new_brand_week is not None:
        appears = (week >= int(new_brand_week)) & (rng.random(n) < 0.20)
        brand[appears] = _NEW_BRAND

    model_pick = rng.integers(0, 3, size=n)
    model = np.empty(n, dtype=object)
    for i in range(n):
        choices = (
            _NEW_BRAND_MODELS if brand[i] == _NEW_BRAND else _BRAND_MODELS[brand[i]]
        )
        model[i] = choices[model_pick[i] % len(choices)]

    title_status = rng.choice(_TITLE_STATUSES, size=n, p=_TITLE_WEIGHTS)
    condition = rng.choice(_CONDITIONS, size=n, p=_CONDITION_WEIGHTS)
    state = rng.choice(_STATES, size=n)
    vin = np.array([f"SYN{i:012d}" for i in range(n)], dtype=object)

    premium = np.array([_BRAND_PREMIUM.get(b, 4000.0) for b in brand])
    price = (
        3000.0
        + premium
        + (year - 1995) * 450.0
        - mileage * 0.04
        + rng.normal(0, 1500, size=n)
    )
    price = (
        price * (1.0 + inflation) ** week
    )  # gradual numeric drift across weeks -> PSI
    if shock_week is not None:
        # INJECTED SCENARIO: a one-time, permanent market price jump at a single week (e.g. a
        # pricing-source change), so PSI trips once at that transition -- distinct from the gradual
        # inflation above, which stays below the alert threshold week to week.
        price = np.where(week >= int(shock_week), price * shock_multiplier, price)
    price = np.clip(price, 800.0, None)

    df = pd.DataFrame(
        {
            "price": price.astype(float),
            "brand": brand,
            "model": model,
            "year": year.astype("int64"),
            "mileage": mileage.astype(float),
            "title_status": title_status.astype(object),
            "vin": vin,
            "condition": condition.astype(object),
            "state": state.astype(object),
            "posting_date": posting_date,
        }
    )

    # Controlled null-rate drift: this column's null fraction climbs week over week.
    if rising_null_col and rising_null_col in df.columns:
        p_null = np.clip(0.02 + 0.03 * week, 0.0, 0.6)
        df.loc[rng.random(n) < p_null, rising_null_col] = np.nan

    defects = _inject_defects(df, rng, bad_fraction)
    df.attrs["n_rows"] = n
    df.attrs["defects"] = defects
    df.attrs["n_defect_rows"] = int(sum(defects.values()))
    return df


def _inject_defects(
    df: pd.DataFrame, rng: np.random.Generator, bad_fraction: float
) -> dict[str, int]:
    """Corrupt ~bad_fraction of rows with the SPEC 6.3 defects; return per-defect counts."""
    n = len(df)
    defects = {
        "null_brand": 0,
        "nonpositive_price": 0,
        "impossible_year": 0,
        "duplicate_vin": 0,
        "stale_date": 0,
    }
    n_bad = min(int(round(float(bad_fraction) * n)), n)
    if n_bad <= 0:
        return defects

    bad_rows = rng.choice(n, size=n_bad, replace=False)
    null_idx, price_idx, year_idx, vin_idx, date_idx = np.array_split(bad_rows, 5)

    if len(null_idx) > 0:
        df.loc[null_idx, "brand"] = None
        defects["null_brand"] = len(null_idx)
    if len(price_idx) > 0:
        bad_price = -rng.uniform(1, 5000, size=len(price_idx))
        bad_price[::2] = 0.0  # a mix of zeros and negatives
        df.loc[price_idx, "price"] = bad_price
        defects["nonpositive_price"] = len(price_idx)
    if len(year_idx) > 0:
        df.loc[year_idx, "year"] = rng.choice(_IMPOSSIBLE_YEARS, size=len(year_idx))
        defects["impossible_year"] = len(year_idx)
    if len(vin_idx) >= 2:  # need at least two rows to form a duplicate
        df.loc[vin_idx, "vin"] = _DUPLICATE_VIN
        defects["duplicate_vin"] = len(vin_idx)
    if len(date_idx) > 0:
        df.loc[date_idx, "posting_date"] = _SYNTHETIC_START - _STALE_OFFSET
        defects["stale_date"] = len(date_idx)
    return defects


def iter_batches(
    df: pd.DataFrame, batching: dict[str, object]
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (label, batch_df) slices ordered by the configured date column.

    Simulates data arriving over time so monitoring has a real run history and drift has a
    previous run to compare against. Slices smaller than batching['min_batch_rows'] are skipped.
    """
    date_col = str(batching["date_column"])
    freq = str(batching["freq"])
    min_rows = int(batching["min_batch_rows"])

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    mask = parsed.notna()
    work = df.loc[mask].copy()
    periods = parsed.loc[mask].dt.to_period(freq)

    for period in sorted(periods.unique()):
        batch = work.loc[periods == period]
        if len(batch) < min_rows:
            continue
        yield _period_label(period, freq), batch.reset_index(drop=True)


def _period_label(period: pd.Period, freq: str) -> str:
    """Human-readable batch label; ISO year-week (e.g. '2024-W03') for weekly frequencies."""
    if freq.upper().startswith("W"):
        iso = period.start_time.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return str(period)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic used-car sample CSV."
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="generate synthetic data"
    )
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = generate_synthetic(n=args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    _cli()
