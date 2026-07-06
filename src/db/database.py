"""SQL persistence via SQLAlchemy. SQLite by default; Postgres optional (creds from .env).

Only *validated* data should be written here -- the data-quality gate runs first (pipeline.py),
and the --load CLI re-runs that gate before writing. Credentials come only from the environment
(via src.config), never from source.

CLI:
    python -m src.db.database --load
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine

from src.config import load_settings

log = logging.getLogger("db.database")


def get_engine() -> Engine:
    """Return a SQLAlchemy engine from Settings.db_url (SQLite local/CI, Postgres from env)."""
    url = load_settings().db_url
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def write_df(
    df: pd.DataFrame,
    table: str,
    engine: Engine | None = None,
    if_exists: str = "replace",
) -> int:
    """Write a DataFrame to `table` and return the row count; backend-agnostic (SQLite/Postgres).

    Defaults to `if_exists="replace"` so a re-run of `make all` is idempotent; the pipeline can pass
    "append" to accumulate validated batches during a backfill.
    """
    engine = engine if engine is not None else get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    log.info("wrote %d rows to table %r (%s)", len(df), table, if_exists)
    return len(df)


def read_df(table_or_query: str, engine: Engine | None = None) -> pd.DataFrame:
    """Read a table name or a SQL query into a DataFrame."""
    engine = engine if engine is not None else get_engine()
    return pd.read_sql(table_or_query, engine)


def _cli() -> int:
    """CLI: validate the configured dataset and, only if the DQ gate passes, write it to SQL."""
    parser = argparse.ArgumentParser(description="Load validated data into SQL.")
    parser.add_argument(
        "--load", action="store_true", help="validate, then write the dataset to SQL"
    )
    args = parser.parse_args()
    if not args.load:
        parser.print_help()
        return 0

    from src.ingest.dataset import load as load_data
    from src.quality.report import run_report

    settings = load_settings()
    df = load_data(settings)
    report = run_report(df, settings)
    if not report.passed:
        print("DATA QUALITY GATE FAILED -- refusing to write unvalidated data.")
        return 1
    n = write_df(df, settings["database"]["table"], get_engine())
    print(f"loaded {n} validated rows -> {settings['database']['table']}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
