# used-car-price-pipeline

A containerized, tested data pipeline that ingests used-car listings in **time-ordered batches**,
validates them through a **two-layer data-quality system** (pandera schema + business rules),
stores them in SQL, **monitors data quality and drift across runs**, trains a price **regressor**
with an honestly measured metric, and surfaces it all on a **Streamlit dashboard**.

> **Status:** scaffold + spec complete; implementation in progress. All results below are `TODO`
> placeholders until produced by a real run — see [SPEC.md](SPEC.md) §2 (honesty).

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
                                          └ halts on ERROR        └────────── Streamlit dashboard ──────────┘
```

- **Two-layer data quality** — a **pandera** schema (technical: columns, dtypes, ranges) plus
  hand-rolled **business rules** (category membership, cross-field consistency, duplicates). An
  `ERROR` halts the pipeline before bad data becomes a model. Every rule is unit-tested against
  clean **and** deliberately broken data.
- **Monitoring & drift** — each batch is recorded to a `runs` table with its DQ pass-rate, key
  column stats, and (after training) model error. **PSI**, null-rate, category-shift, and
  **freshness** are compared run-over-run and alerted on.
- **Dashboard** — a **Streamlit** app plots quality, drift, and MAE across runs.
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
in-memory, so `make all`, `make backfill`, and `docker compose up` work out of the box. To use a
real dataset instead, set `dataset.source` in `config/config.yaml` and run `make data-real`.

## Dataset

By default the pipeline uses a **self-contained synthetic generator** (`dataset.source: synthetic`)
that produces a realistic, messy used-car dataset with **known** injected defects and controlled
drift across weeks -- no download, no account, reproducible, and ideal ground truth for a
data-quality project. Optionally point at a real public dataset (`url` / `kaggle` / `openml`) with
`make data-real`; raw data is never committed (`data/raw/` is git-ignored). See `SECURITY.md`.

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

Per run, tracked in the `runs` table and plotted on the dashboard: DQ pass-rate, rows ingested,
data **freshness**, model **MAE**, and drift signals — **PSI** on price/mileage/year, null-rate
deltas, and category shift — vs. the previous run.

## Results

_Populated from the real run by `make evaluate` / `make backfill` — do not hand-edit._

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Mean baseline | TODO | TODO | TODO | TODO |
| Linear Regression | TODO | TODO | TODO | TODO |
| Decision Tree | TODO | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO | TODO |

**Data-quality catch rate:** _TODO_  ·  **Drift alerts across runs:** _TODO_  ·  _(dashboard screenshot: TODO)_

## Project layout

See [SPEC.md](SPEC.md) §5 for the full tree + per-module responsibilities and [CLAUDE.md](CLAUDE.md)
for how the project is built.

## Honest limitations

- Trained by default on a synthetic (but realistically messy) dataset; not a production pricing model. Use `make data-real` for a real-world MAE.
- "Data over time" is simulated by partitioning the dataset by `posting_date` (documented design),
  not a live feed.
- Live scraping (`src/ingest/scraper.py`) is optional and off by default.

## License

TODO (e.g. MIT). Dataset retains its own license — see **Dataset**.
