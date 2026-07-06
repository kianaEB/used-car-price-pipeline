# SPEC — used-car-price-pipeline

**One-liner:** A containerized, reproducible, tested data pipeline that ingests used-car listings
in time-ordered batches, validates them through a two-layer **data-quality** system (declarative
schema + business rules), stores them in SQL, **monitors data quality and drift across runs**,
trains a price **regressor** with an honestly measured metric, and surfaces it all on a **quality
dashboard**.

This is the design contract. Build it in the order in `CLAUDE.md`. The data-quality **and
monitoring** layers plus the tests are the point of the project — not the model, and not
decoration. If `SPEC.md` and `CLAUDE.md` ever conflict, flag it rather than guessing.

---

## 1. Why this project exists

Ground-up rebuild of an old student script (`reference/cars_original.py`) that scraped cars.com,
dumped rows into MySQL, and "predicted" price with a classifier. Fixing its flaws *is* the story,
but we go further: this is built to look like the data-quality / data-platform work a real team
does, so it stands on its own across Data QA, Data Engineering, and Data Science applications.

| Prototype flaw | The rebuild |
|---|---|
| Classifier on a continuous price | Proper **regressor** + baselines, real error metrics |
| No split, **no metric** | Held-out split; **MAE / RMSE / R² / MAPE**, tracked across runs |
| Mileage silently zeroed | Correct handling + a DQ rule that would catch it |
| No validation of the data | **Two-layer DQ** (pandera schema + business rules) with a hard gate |
| One-shot, no history | **Run history + drift/freshness monitoring** across time-ordered batches |
| Hardcoded DB password | **Secrets via env vars**; real **Postgres** in Docker, SQLite for tests |
| One 207-line script, no tests | Modular package, **pytest** + CI **coverage gate**, `docker compose up` |
| Numbers you couldn't see | **Streamlit dashboard**: quality + drift + model metrics over time |

**Portfolio goals (priority order):**

1. **Be a credible data-quality & monitoring project** — the thing a QA / data-platform team
   recognizes: schema contracts, business rules, a gate, and **drift/freshness monitoring over
   time**. (Directly targets trivago's *Core Data QA* role — "monitor systems and validate the
   accuracy and reliability of data".)
2. **Show honest ML** — a correct regression setup, honest baselines, a real reproducible metric,
   no leakage, tracked run-over-run.
3. **Be reusable and defensible** — clean enough to headline on multiple résumés and to explain,
   line by line, in a technical interview. Every capability must be one you can defend.

**Anti-goals:** not a Kaggle-score chase, not a distributed-systems showcase. No Kafka, no
Kubernetes, no cloud deployment, no Spark, no feature store — those are over-reach for this role
and hard to defend. Depth on quality + monitoring beats breadth of buzzwords.

---

## 2. Non-negotiable principles

1. **Honesty.** Every number in `README.md`, the dashboard screenshots, or a résumé must come
   from a real run. No hand-typed metrics. Unfilled → `TODO`.
2. **Reproducibility.** `docker compose up` (or `make all` in a venv) reproduces every artifact
   and metric from a clean checkout after the documented data download. Seeds fixed and threaded.
3. **Tested, with a gate.** DQ checks, drift math, and metrics are covered by `pytest`; CI runs
   the suite and enforces a **coverage gate** (see §12). Green CI is part of Done.
4. **No secrets in git.** DB credentials come from the environment (`.env`, git-ignored).
   `.env.example` documents names only. Compose reads them from `.env`.
5. **Deterministic, network-optional core.** `make all`, tests, and CI run offline from a local
   file or the synthetic generator. Live scraping stays an optional, off-by-default module.

---

## 3. Scope

**In scope**

- Ingest a **large, genuinely messy** used-car dataset, sliced into **time-ordered batches**
  (simulating data arriving over time) plus a synthetic generator for tests/CI.
- **Two-layer data quality:** a **pandera** declarative schema (technical: columns, dtypes,
  nullability, ranges) + hand-rolled **business rules** (category membership, cross-field
  consistency), producing a structured report and a **hard gate**.
- **Monitoring:** persist a `RunRecord` per batch (row counts, DQ pass rate, key column stats,
  freshness, model metrics); compute **drift** (PSI + null-rate + category shift) and
  **freshness** versus the previous run; flag alerts.
- **SQL storage:** SQLAlchemy; **Postgres** via Docker Compose (default in the container), SQLite
  for local/CI. Only validated data is written.
- **Modeling:** a price **regressor** + honest baselines; MAE/RMSE/R²/MAPE on a held-out split,
  recorded per run.
- **Dashboard:** a **Streamlit** app reading the run history — quality pass-rate, drift (PSI),
  freshness, row counts, and model MAE over runs, plus the latest DQ report.
- **Packaging:** `Dockerfile` + `docker-compose.yml` (db + pipeline + dashboard); `Makefile`;
  `pytest` + GitHub Actions CI with a coverage gate.

**Optional / stretch (only after the above is green)**

- `src/ingest/scraper.py`: a fixed, rate-limited, robots-respecting cars.com scraper to refresh
  data. Off by default; never on the `make all` / test / CI path.
- Simple hyperparameter search; SHAP or permutation feature importance in the dashboard.
- `ANALYSIS.md` write-up (what the DQ layer caught; how drift behaved; model comparison).

**Out of scope (do not add):** Kafka/streaming, Kubernetes, cloud deploy, Spark, a feature store,
deep learning, a multi-service microservice mesh.

---

## 4. Architecture & data flow

```
 raw slice (by posting_date)
        │
        ▼
   ingest.load ─▶ QUALITY (2 layers) ───────────────┐
        │          1. pandera schema  (technical)    │  fail ERROR ⇒ STOP
        │          2. business rules  (consistency)  │  (no DB, no train)
        │                                            ▼
        │                                   quality/report.json
        ▼
   monitoring.record_run ──▶ runs table (SQL) ──▶ monitoring.drift(prev, curr)
        │                                            │  PSI / null-rate / freshness
        ▼                                            ▼
   db.write_df (validated) ─▶ features ─▶ train ─▶ evaluate ─▶ metrics (also recorded to the run)
                                                             │
                    dashboard (Streamlit) ◀── reads runs table + latest report
```

`src/pipeline.py` runs one batch: **ingest → validate (gate) → record run → drift vs previous →
load → features → train → evaluate → update run with metrics**. `make backfill` replays all
batches in `posting_date` order to build a real run history the dashboard can plot.

---

## 5. Directory layout

```
used-car-price-pipeline/
├── README.md  SPEC.md  CLAUDE.md
├── requirements.txt  .gitignore  .env.example  pytest.ini  Makefile
├── Dockerfile  docker-compose.yml  .dockerignore
├── config/config.yaml            # paths, db, batching, DQ thresholds, drift thresholds, model
├── data/{raw,interim,processed}/ # raw git-ignored; .gitkeep keeps folders
├── src/
│   ├── config.py                 # YAML + .env → typed Settings (SQLite/Postgres URL)
│   ├── ingest/
│   │   ├── dataset.py            # load CSV → canonical schema; time-slice batches; synthetic gen
│   │   └── scraper.py            # OPTIONAL polite scraper (stretch; off by default)
│   ├── quality/
│   │   ├── schema.py             # pandera DataFrameSchema (technical layer)
│   │   ├── checks.py             # business rules → CheckResult (consistency, categories, dupes)
│   │   └── report.py             # run schema + rules → DataQualityReport; JSON; hard gate
│   ├── monitoring/
│   │   ├── runs.py               # RunRecord dataclass; save/load run history to SQL
│   │   └── drift.py              # PSI, null-rate delta, category shift, freshness vs previous run
│   ├── db/database.py            # SQLAlchemy engine, write_df / read_df (SQLite + Postgres)
│   ├── features/preprocess.py    # clean, encode, leakage-safe split
│   ├── model/{train.py,evaluate.py}
│   └── pipeline.py               # orchestrate one batch; backfill replays all
├── dashboard/app.py              # Streamlit: quality + drift + metrics over runs
├── tests/                        # pytest: quality, drift/monitoring, preprocess, db, model
├── notebooks/exploration.ipynb   # optional EDA
├── reference/cars_original.py    # preserved prototype (password redacted) — do not run
└── .github/workflows/ci.yml      # lint + pytest + coverage gate (offline, synthetic)
```

---

## 6. Data

### 6.1 Data source (synthetic by default; real dataset optional)

Use a **large, messy, real** used-car dataset so validation and drift have genuine signal.
**Reference choice: the "Craigslist Used Cars" dataset** (~400k rows; columns include `price`,
`year`, `manufacturer`, `model`, `condition`, `odometer`, `title_status`, `vin`, `state`,
`posting_date`, …). It is genuinely dirty — zero/absurd prices, missing years, null-heavy
columns, duplicate VINs, and a real `posting_date` for **freshness/drift** — which is exactly what
makes the quality layer worth building. Any comparable large public used-car CSV is fine if
documented in `README.md`.

**The default `dataset.source: synthetic` needs no download** (see 6.3) -- the whole pipeline runs
offline, no account. For a real run, set `dataset.source` to `url`/`kaggle`/`openml` and run
`make data-real`, which caches into `data/raw/vehicles.csv` (git-ignored); `README.md` then states
the dataset, source, license, and row count. Never commit raw data.

### 6.2 Time-ordered batches (simulate data over time)

`ingest.iter_batches(df, freq)` yields slices of the data ordered by `posting_date` (e.g. weekly).
Each slice is one pipeline run, which is what gives monitoring a real **run history** and drift a
**previous run** to compare against. This is an honest, defensible design — document it as
"partitioned by posting date to simulate incremental loads."

### 6.3 Synthetic generator (DEFAULT source; also tests + CI)

`dataset.generate_synthetic(n, seed, bad_fraction)` returns the canonical schema with realistic
ranges and a controllable fraction of injected defects (nulls, negative/zero price, impossible
year, duplicate VIN, stale posting_date). Lets tests and CI run offline against data with **known**
defects and lets drift tests compare two synthetic slices with a **known** shift.

### 6.4 Canonical schema (contract — enforced by pandera in `quality/schema.py`)

| column | type | rule |
|---|---|---|
| `price` | float | > 0, ≤ configurable max |
| `brand` | str | non-null (maps from `manufacturer`) |
| `model` | str | nullable-tolerant; report if too many nulls |
| `year` | int | 1950 ≤ year ≤ current_year + 1 |
| `mileage` | float | ≥ 0, < configurable max (maps from `odometer`) |
| `title_status` | str | in a known set (else WARN) |
| `vin` | str | unique if present (ERROR on dupes) |
| `posting_date` | datetime | parseable; used for batching + freshness |

Missing optional columns must not crash the pipeline; `config.yaml` declares which checks are active.

---

## 7. Data-quality layer (two layers)

**Layer 1 — technical schema (`quality/schema.py`, pandera).** A declarative
`pandera.DataFrameSchema` (or `DataFrameModel`) encoding columns, dtypes, nullability, and numeric
ranges. Validate with `lazy=True` so *all* failures are collected, then translate pandera's
failure cases into `CheckResult`s. This is the "technical data-quality checks" half of the JD.

**Layer 2 — business rules (`quality/checks.py`).** Hand-rolled rules that a schema can't express,
each returning a `CheckResult` (see dataclass below): category membership (`title_status`),
cross-field **consistency** (e.g. "like new" with very high odometer), duplicate rows/VINs, and
min-row/non-empty. This is the "business data-quality checks" half.

```python
@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: Literal["ERROR", "WARN", "INFO"]
    n_violations: int
    n_rows: int
    detail: str = ""
    sample: list[dict] = field(default_factory=list)
```

**`quality/report.py`** runs Layer 1 then Layer 2, assembles a `DataQualityReport`, prints a
table, writes `data/processed/dq_report.json`, and enforces the **gate**: any `ERROR`-severity
failure stops the pipeline before monitoring-record/DB-load/training. `WARN`s are logged.

---

## 8. Monitoring (the standout)

**`src/monitoring/runs.py`** — a `RunRecord` (run id, timestamp, batch label, n_rows, dq_pass_rate,
n_error_checks, per-key-column stats {mean, null_rate}, freshness_days, and — after training —
model MAE/RMSE/R²). `save_run(record, engine)` appends to a `runs` table; `load_runs(engine)`
returns the history as a DataFrame; `latest_run` / `previous_run` helpers.

**`src/monitoring/drift.py`** — compares the current batch to the previous run:

- `psi(expected, actual, bins)` — **Population Stability Index** on numeric columns (price,
  mileage, year). Threshold in config (e.g. PSI > 0.2 ⇒ drift alert). *Implement this helper; it is
  unit-tested with a known no-drift (≈0) and known-shift (large) case.*
- `null_rate_delta` — change in per-column null fraction beyond a threshold.
- `category_shift` — top-category share change for `title_status` / `brand`.
- `freshness` — age of the newest `posting_date` vs a max-staleness threshold.
- Returns a `DriftReport` (per-signal value + alert flag); the pipeline logs alerts and the
  dashboard plots them. Drift does **not** hard-stop the pipeline (it's a signal, not a gate) —
  document that choice.

---

## 9. Modeling & metrics

Same as a correct regression setup: features `brand, model, year, mileage` (+ available
categoricals); encoders **fit on train only**; `train_test_split` from config. Compare **mean
baseline, LinearRegression, DecisionTreeRegressor, RandomForestRegressor**; score **MAE, RMSE,
R², MAPE** on the held-out set; save all to `artifacts/metrics.json` and record the winner's
metrics onto the run. Report honestly; the winner must beat the mean baseline or the README says so.

---

## 10. Database & packaging

- **SQLAlchemy**; backend from `.env`/config: **Postgres** (Docker default) or **SQLite** (local/CI).
  Credentials only from env. `database.py`: `get_engine()`, `write_df()`, `read_df()`.
- **Docker:** `Dockerfile` (python:3.11-slim) for the pipeline/dashboard image; `docker-compose.yml`
  with services **`db`** (postgres:16, env from `.env`, named volume), **`pipeline`** (runs
  `make all` against `db`), and **`dashboard`** (Streamlit on port 8501). `docker compose up`
  brings the whole thing up. `.dockerignore` excludes data/raw, .venv, .git, artifacts.

---

## 11. Testing strategy

- Fixtures (`tests/conftest.py`): clean + dirty synthetic frames; two slices with a known drift.
- `test_quality_checks.py` — each business rule passes on clean data, fails on its defect, right
  severity + `n_violations`.
- `test_schema.py` — the pandera schema accepts clean data and raises/【collects】 on schema
  violations.
- `test_monitoring.py` — **`psi()` ≈ 0 for identical distributions and large for a known shift**;
  `record_run`/`load_runs` round-trip; freshness math.
- `test_preprocess.py` — deterministic split; no encoder leakage.
- `test_database.py` — `write_df` → `read_df` round-trip (SQLite temp).
- `test_model.py` — metrics computed; winner beats mean baseline on synthetic data; deterministic.
- Green locally + in CI; **coverage gate** enforced (see §12), highest on `src/quality` + `src/monitoring`.

---

## 12. Definition of Done (acceptance checklist)

- [ ] `docker compose up` brings up Postgres + runs the pipeline + serves the dashboard; **and**
      `make all` works in a plain venv (SQLite) with the default synthetic source (no download).
- [ ] `make backfill` replays time-ordered batches and produces a **runs table with ≥ 3 runs**;
      the dashboard plots quality + drift + MAE over those runs.
- [ ] `make test` green locally and in CI; CI enforces **`--cov-fail-under=70`** (raise from the
      starter value as modules land). Coverage on `src/quality` + `src/monitoring` is the priority.
- [ ] An `ERROR`-severity DQ failure demonstrably halts the pipeline before DB load / training.
- [ ] A **known distribution shift raises a drift alert**; identical data does not.
- [ ] `artifacts/metrics.json` has MAE/RMSE/R²/MAPE for all models; the winner beats the baseline.
- [ ] No secrets in git; `.env` ignored; `.env.example` lists names only; Postgres creds from env.
- [ ] `README.md` results + dashboard screenshot filled **from a real run** (no placeholder numbers);
      dataset + license + row count stated.
- [ ] `reference/cars_original.py` present, not on any run path, no live secret.
- [ ] Typed + docstringed; `ruff`/`black` clean.