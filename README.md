# used-car-price-pipeline

A containerized, tested data pipeline that ingests used-car listings in **time-ordered batches**,
validates them through a **two-layer data-quality system** (pandera schema + business rules),
stores them in SQL, **monitors data quality and drift across runs**, trains a price **regressor**
with an honestly measured metric, and surfaces it all on a **Streamlit dashboard**.

> **Status:** built and tested (pytest + a CI coverage gate, `docker compose up --build`). The
> results below are from a **real** Kaggle Craigslist run; the committed default stays
> **synthetic/offline** so CI, `make all`, and the container run with no network.

<!-- Live once on GitHub with CI enabled:
[![ci](https://github.com/kianaEB/used-car-price-pipeline/actions/workflows/ci.yml/badge.svg)](../../actions)
-->

## Why

A ground-up rebuild of an old student script ([`reference/cars_original.py`](reference/cars_original.py))
that used a *classifier* to predict a continuous price, never split or evaluated the data, silently
zeroed out mileage, and hardcoded a DB password. The rebuild fixes each of those — and makes **data
quality and monitoring** the centerpiece, which is the work a real data-platform / QA team does.

## What it does

```
raw slice (by posting_date) ─▶ ingest ─▶ QUALITY (2 layers) ─▶ record run ─▶ drift vs prev ─▶ SQL ─▶ features ─▶ model ─▶ metrics
                                          1. pandera schema       │ (run history)   │ PSI/nulls/freshness
                                          2. business rules       │                 │
                                          └ ERROR ⇒ quarantine    └────────── Streamlit dashboard ──────────┘
```

- **Two-layer data quality** — a **pandera** schema (technical: columns, dtypes, ranges) plus
  hand-rolled **business rules** (category membership, cross-field consistency, duplicates). The
  gate **quarantines** the exact rows the DQ layer flags and **hard-halts** a batch that is empty,
  missing columns, or above a configurable error ceiling. Every rule is unit-tested against clean
  **and** deliberately broken data.
- **Monitoring & drift** — each batch is recorded to a `runs` table with its DQ pass-rate, key
  column stats, quarantine count, freshness, and (after training) model error. **PSI**, null-rate,
  category-shift, and **freshness** are compared run-over-run and alerted on.
- **Dashboard** — a **Streamlit** app plots quality, drift, and model error across runs.
- **SQL** — SQLAlchemy; **Postgres** in Docker Compose, SQLite for local/CI.
- **Honest modeling** — a proper regressor (Decision Tree / Random Forest) vs. a mean baseline and
  linear regression, scored with MAE / RMSE / R² / MAPE on a held-out split, tracked per run.
- **Reproducible** — one config file, fixed seeds, `make` targets, `docker compose up`, and CI with
  a coverage gate.

## Quickstart

```bash
# Full stack (Postgres + pipeline + dashboard) — dashboard at http://localhost:8501
cp .env.example .env          # fill locally; never commit
docker compose up --build

# …or run locally in a venv (SQLite):
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows (PowerShell/CMD)
pip install -r requirements.txt
make data        # small synthetic sample (offline smoke run)
make backfill    # replay time-ordered batches → build run history
make dashboard   # Streamlit at http://localhost:8501
make test        # pytest (+ coverage)
```

> **On Windows:** run these from **Git Bash** or **WSL** so `make`, `cp`, and `source` work as
> written (plain PowerShell/CMD has no `make` — either use Git Bash, `choco install make`, or call
> the `python -m ...` commands directly; see the Windows note in `CLAUDE.md`).

No download needed: the default `dataset.source: synthetic` generates a realistic, messy dataset
in-memory, so `make all`, `make backfill`, and `docker compose up` work out of the box. To run on
the real dataset instead, see **[Dataset](#dataset)** — it is an env-driven opt-in and the committed
config stays synthetic/offline.

## Dataset

By default the pipeline uses a **self-contained synthetic generator** (`dataset.source: synthetic`)
— a realistic, messy used-car dataset with **known** injected defects and **controlled drift** (a
one-time price shock, a new brand mid-stream, a rising null-rate column), needing no download or
account. It is the committed default, so `make all`, `make backfill`, CI, and `docker compose up`
all run **offline** with exact ground truth.

The **Results** below come from a real public dataset:
[**Kaggle `austinreese/craigslist-carstrucks-data`**](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
— **426,880** used-car listings (**426,812** with a valid `posting_date`), licensed **CC0-1.0**,
scraped **2021-04-04 → 2021-05-05**. Raw data is never committed (`data/raw/` is git-ignored).

The real dataset's freshness reference date and error ceiling differ from synthetic, so they are
passed as **env overrides** (the committed config is untouched):

```bash
# 1. Fetch the CSV (needs a Kaggle account + ~/.kaggle/kaggle.json):
DATASET_SOURCE=kaggle make data-real            # -> data/raw/vehicles.csv (git-ignored)

# 2. Reproduce the real backfill:
DATASET_SOURCE=kaggle \
FRESHNESS_REFERENCE_DATE=2021-05-06 \
QUALITY_MAX_ERROR_FRACTION=0.70 \
  python -m src.pipeline --backfill
```

## Data-quality checks

| Layer | Check | Catches | Severity |
|---|---|---|---|
| schema (pandera) | columns / dtypes / ranges / nullability | wrong types, negative/implausible price·year·mileage, missing required cols | ERROR |
| business | min_rows / not_empty | truncated or empty ingest | ERROR |
| business | duplicates | duplicated rows; repeated VIN | ERROR / WARN |
| business | categories | unknown title_status | WARN |
| business | consistency | e.g. "like new" with very high odometer | WARN |

Full run summary → `data/processed/dq_report.json`.

## Monitoring

Per run, tracked in the `runs` table and plotted on the dashboard: DQ pass-rate, rows ingested vs
quarantined, data **freshness**, model **MAE/RMSE**, and drift signals — **PSI** on
price/mileage/year, null-rate deltas, and category shift — vs. the previous run.

## Results

_Real run on the Kaggle Craigslist dataset (see [Dataset](#dataset)); every number below is pulled
from `artifacts/metrics.json` and the `runs` table — not hand-edited._

**Model comparison** — held-out test split of the final batch (`2021-W18`):

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Mean baseline | 10,207 | 14,174 | -0.000 | 51.48 |
| Linear Regression | 7,274 | 11,388 | 0.354 | 61.38 |
| Decision Tree | 6,492 | 10,954 | 0.403 | 49.14 |
| **Random Forest** | **5,833** | **9,736** | **0.528** | 53.06 |

**Random Forest wins** — MAE **5,833** vs. the mean baseline's **10,207**, a **~43% improvement**
(R² **0.53** on genuinely messy real prices). MAPE runs high because a few sub-$100 listings survive
the price gate and blow up the percentage error, so **MAE/RMSE are the honest headline**.

**Data-quality catch-rate: 43.0%** — **183,348 of 426,812** rows flagged and quarantined across the
6 weekly batches, dominated by **duplicate / reposted VINs** (~33% of rows), plus **≤ 0 or absurd
prices** (many `$0`; one at **$3.7 billion**), **impossible years** (`< 1950`), and **null brand**.
The model trains only on the validated remainder.

**Drift: 1 alert across 6 runs** — a null-rate jump on `vin` at `2021-W14`. Real ~30-day
distributions are otherwise stable, so PSI/category alerts stay quiet — the *controlled* drift demo
(a one-time price shock → a single PSI trip, a new brand → category shift, a rising null-rate) lives
in the default **synthetic** mode.

![Streamlit dashboard — DQ pass-rate, PSI, quarantine counts, freshness, and model error over runs](docs/dashboard.png)

## Project layout

See [SPEC.md](SPEC.md) §5 for the full tree + per-module responsibilities and [CLAUDE.md](CLAUDE.md)
for how the project is built.

## Honest limitations

- **The real-data gate is deliberately generous.** Real Craigslist batches are **33–58%
  quarantinable**, so the real run raises the error ceiling to **0.70** via
  `QUALITY_MAX_ERROR_FRACTION` (env-driven); the gate still hard-halts any batch above 70%. The
  committed **synthetic** ceiling stays a strict **0.25**.
- **Duplicate VINs are dropped, not deduplicated.** `check_duplicates` flags *all* copies of a
  repeated VIN (`keep=False`), so every repost of the same car is quarantined rather than collapsed
  to a single row — simpler and safe, but it discards data a dedup step would keep.
- **High-cardinality `model` is top-N capped.** The real `model` column has ~30k free-text values;
  the one-hot encoder caps categories (`features.max_categories`, top-N + an "infrequent" bucket)
  so the feature matrix doesn't explode — a modeling simplification, and a no-op for synthetic.
- **"Data over time" is simulated** by partitioning on `posting_date` (a documented design), not a
  live feed. Live scraping (`src/ingest/scraper.py`) is optional and off by default.
- Trained on public listing data — **not** a production pricing model.

## License

**Dataset:** Kaggle `austinreese/craigslist-carstrucks-data` is licensed **CC0-1.0** and retains its
own terms; raw data is not redistributed here. **Code:** license at the maintainer's discretion
(MIT recommended — add a `LICENSE` file to finalize).
