"""Load the used-car dataset for the pipeline.

DEFAULT source is 'synthetic' (config: dataset.source) -- the pipeline runs end to end with NO
download and NO accounts. `load(settings)` dispatches on the configured source. A real dataset is
optional (see src/ingest/download.py and `make data-real`).

CLI:
    python -m src.ingest.dataset --synthetic --out data/raw/sample.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = [
    "price", "brand", "model", "year", "mileage", "title_status", "vin", "posting_date",
]


def load(settings) -> pd.DataFrame:
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


def load_dataset(path, column_map: dict | None = None) -> pd.DataFrame:
    """Read a real CSV and normalize columns to the canonical schema.

    TODO:
      - read CSV; rename via column_map (manufacturer->brand, odometer->mileage)
      - lower/strip column names; coerce dtypes WITHOUT dropping bad rows (the DQ layer reports them)
    """
    raise NotImplementedError


def generate_synthetic(n=8000, seed=42, bad_fraction=0.06, n_weeks=8, drift=None) -> pd.DataFrame:
    """Return a realistic, MESSY synthetic used-car dataset with KNOWN ground truth.

    Why synthetic-by-default: it makes the whole project self-contained (no download / account),
    reproducible, and -- crucially for a data-QUALITY project -- gives KNOWN defect and drift
    ground truth, so the DQ catch-rate and drift alerts are exactly measurable.

    Must produce, deterministically under `seed`:
      - plausible price ~ f(year, mileage, brand) + noise; realistic brand/model/title_status
      - a `posting_date` spread across `n_weeks` weekly batches (so monitoring has run history)
      - injected defects at rate `bad_fraction` (nulls, negative/zero price, impossible year,
        duplicate VIN) -- return/record their count so the README catch-rate is exact
      - controlled DRIFT across weeks from `drift`: price_inflation_per_week (numeric drift -> PSI),
        rising null-rate in `rising_null_column`, and a `new_brand_week` (category shift)

    TODO: implement per the contract above.
    """
    raise NotImplementedError


def iter_batches(df, batching: dict):
    """Yield (label, batch_df) slices ordered by the configured date column.

    Simulates data arriving over time so monitoring has a real run history and drift has a
    previous run to compare against.

    TODO: parse batching['date_column'] to datetime; group by a pandas period at
    batching['freq'] (e.g. 'W'); yield ("2021-W03", slice) chronologically; skip slices
    smaller than batching['min_batch_rows'].
    """
    raise NotImplementedError


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic used-car sample CSV.")
    parser.add_argument("--synthetic", action="store_true", help="generate synthetic data")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = generate_synthetic(n=args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    _cli()
