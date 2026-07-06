.PHONY: setup data data-real validate db features train evaluate all backfill dashboard test lint clean

PY = python

setup:
	pip install -r requirements.txt

# generate the synthetic dataset (DEFAULT source; no download, no accounts)
data:
	$(PY) -m src.ingest.dataset --synthetic --out data/raw/sample.csv

# OPTIONAL: fetch a real dataset per config/config.yaml `dataset.source` (url/kaggle/openml)
data-real:
	$(PY) -m src.ingest.download

# run data-quality checks (pandera schema + business rules); exits non-zero on ERROR severity
validate:
	$(PY) -m src.quality.report

# load validated data into SQL
db:
	$(PY) -m src.db.database --load

# clean, encode, train/test split
features:
	$(PY) -m src.features.preprocess

# fit regressor + baselines, save the model
train:
	$(PY) -m src.model.train

# held-out metrics -> artifacts/metrics.json
evaluate:
	$(PY) -m src.model.evaluate

# one batch: ingest -> validate(gate) -> record run -> drift -> db -> features -> train -> evaluate
all:
	$(PY) -m src.pipeline

# replay time-ordered batches to build the run history the dashboard plots
backfill:
	$(PY) -m src.pipeline --backfill

# monitoring dashboard (quality + drift + metrics over runs)
dashboard:
	streamlit run dashboard/app.py

test:
	pytest

lint:
	ruff check src tests && black --check src tests

clean:
	rm -rf artifacts data/processed/*.db data/processed/*.json data/interim/* .pytest_cache
