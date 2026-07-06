# CLAUDE.md — build guide for `used-car-price-pipeline`

You are building this repo from the scaffold. **Read `SPEC.md` fully first.** This file is *how to
work*; `SPEC.md` is *what to build*. If they conflict, flag it — don't guess.

---

## Mission

Turn this scaffold into a genuinely good, defensible, portfolio-grade project: a **containerized,
monitored, tested data-quality pipeline** for used-car listings, with an honestly measured price
regressor downstream. The centerpiece is the **two-layer data quality + drift/freshness
monitoring**, surfaced on a dashboard. It rebuilds the prototype in `reference/cars_original.py` —
the before/after is part of the story.

## Golden rules (do not violate)

1. **Never fabricate results.** Every metric, drift number, or catch-rate in `README.md`, the
   dashboard, or commit messages must come from a real run. No hand-typed numbers. Not run yet ⇒
   leave the `TODO`.
2. **Never commit secrets.** DB credentials come from environment variables only. `.env` is
   git-ignored; `.env.example` holds names with example/empty values; Compose reads `.env`. If a
   real secret ever appears in a diff, stop and remove it.
3. **Tests must pass, and the gate must be real.** `make test` green is part of Done; keep CI's
   coverage gate on. An `ERROR`-severity DQ check must halt the pipeline before monitoring-record,
   DB load, or training — never downgrade it to make a run "succeed."
4. **Every capability must be defensible.** Add nothing you couldn't explain in an interview. If a
   piece feels like résumé-padding you don't understand, cut it. Depth over breadth.
5. **Offline-first core.** `make all`, `make test`, and CI run with no network — from the local
   dataset or the synthetic generator. Scraping is optional and never on the default path.

## How to work — build order (test each stage before moving on)

1. `src/config.py` — settings + DB URL (SQLite local, Postgres from env). Test.
2. `src/ingest/dataset.py` — CSV → canonical schema; `iter_batches`; `generate_synthetic`. Test.
3. **`src/quality/schema.py` (pandera) + `src/quality/checks.py` (business rules) + `report.py`
   (gate)** — the priority. Test each rule against clean and dirty fixtures; test the pandera
   schema accepts clean / collects failures on dirty.
4. **`src/monitoring/runs.py` + `drift.py`** — `RunRecord` save/load round-trip; implement and
   test `psi()` (≈0 for identical, large for a known shift); null-rate/category/freshness. This is
   the standout — make it solid and well-tested.
5. `src/db/database.py` — SQLAlchemy round-trip on SQLite temp. Test.
6. `src/features/preprocess.py` — leakage-safe split + encoders. Test.
7. `src/model/train.py` + `evaluate.py` — regressor + baselines + metrics. Test.
8. `src/pipeline.py` — wire one batch: ingest → validate(gate) → record run → drift → load →
   features → train → evaluate → update run. Add a `backfill` that replays all batches.
9. `dashboard/app.py` — Streamlit over the runs table (quality, drift/PSI, freshness, MAE over runs).
10. `Dockerfile` + `docker-compose.yml` — `docker compose up` = Postgres + pipeline + dashboard.
11. Run for real: `make backfill` (≥3 runs), fill `README.md` results + a dashboard screenshot,
    optionally write `ANALYSIS.md`.

Small, focused commits, each with passing tests. Suggested branch: `build-pipeline`. Conventional
commits (`feat:`, `test:`, `fix:`, `docs:`). Prefer editing the provided stubs over new structure.

## Environment & commands

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows (PowerShell/CMD)
pip install -r requirements.txt
cp .env.example .env            # fill locally; NEVER commit .env

make data        # small synthetic sample for an offline smoke run
make all         # one batch: ingest → validate → monitor → db → features → train → evaluate
make backfill    # replay time-ordered batches to build run history
make dashboard   # streamlit run dashboard/app.py
make test        # pytest (+ coverage)
make lint        # ruff + black --check

# Full stack (Postgres + pipeline + dashboard):
docker compose up --build
```

> **Windows note.** Activate the venv with `.venv\Scripts\activate`. `make` is not installed on
> Windows by default — run the `make` targets from **Git Bash** or **WSL**, or install make (e.g.
> `choco install make`). Without make, call the underlying command directly: `make test` → `pytest`,
> `make all` → `python -m src.pipeline`, `make backfill` → `python -m src.pipeline --backfill`,
> `make data` → `python -m src.ingest.dataset --synthetic --out data/raw/sample.csv`,
> `make dashboard` → `streamlit run dashboard/app.py` (see the `Makefile` for the rest). After each
> build stage, have Claude Code actually run the tests and show the output — don't trust "tests pass"
> without seeing the real run; the gate and tests are the point of this project.

Real dataset (optional): the default synthetic source needs no download. To use a real one, set
`dataset.source` in `config/config.yaml` (url/kaggle/openml) and run `make data-real`, then `make backfill`.

## Coding conventions

- **Python 3.11+**, full **type hints**, one-line docstring on every public function.
- `pathlib.Path`, not string paths. All params/paths/thresholds come from `config.yaml` via
  `src/config.py` — no magic numbers, no hardcoded paths, no hardcoded credentials.
- `logging`, not `print`, in library code (CLIs may print their final line).
- **No import-time side effects.** Entry points under `if __name__ == "__main__":` or in pipeline
  functions. (The prototype's top-level execution is what we're fixing.)
- Determinism: thread the configured `seed` into every split, model, and synthetic generator.
- `ruff` + `black` clean before each commit.

## Definition of Done

Per stage: typed + docstringed, tests written and green, reads config, `ruff`/`black` clean.
Project: `docker compose up` and `make all` both work; `make backfill` yields ≥3 runs the dashboard
plots; `make test` green with CI coverage gate on; the DQ gate halts on ERROR; a known shift raises
a drift alert (identical data doesn't); `metrics.json` winner beats the mean baseline; no secrets in
git; `README.md` filled from a real run with dataset + license; `reference/cars_original.py` present,
off the run path, no live secret. (Full checklist: `SPEC.md §12`.)

## Guardrails / do-not

- Do **not** add Kafka, Kubernetes, cloud deploy, Spark, or a feature store — out of scope (SPEC §3).
- Do **not** make the pipeline depend on live scraping; `scraper.py` stays optional and must respect
  robots.txt / ToS and rate-limit if implemented.
- Do **not** use a classifier for price. Regression only.
- Do **not** let encoders/scalers see test data before the split (no leakage).
- Do **not** let drift hard-stop the pipeline — it's a signal, not the gate (the DQ ERROR gate is).
- Do **not** invent a dataset or numbers. No real dataset yet ⇒ run on synthetic and label any
  numbers as synthetic in `README.md` until a real run is done.
- Do **not** commit `data/raw/*`, `.env`, `*.db`, or `artifacts/*` (see `.gitignore`).

## When you're done — hand back

1. `docker compose up` and `make backfill` succeed; capture real metrics, the drift behavior, and a
   dashboard screenshot.
2. Fill the `README.md` results table, the dataset/license line, and the screenshot.
3. (Optional) `ANALYSIS.md`: what the DQ layer caught, how drift behaved across batches, honest
   model comparison and limitations.
4. Tell Kiana the real MAE/RMSE/R²/MAPE, the DQ catch-rate, and the drift-alert count so she can
   record **Verified** numbers in `documents/project-inventory.md` + `claims-evidence-log.md`.
   Only then may they go on the résumé/LinkedIn.
