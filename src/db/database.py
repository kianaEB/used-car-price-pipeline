"""SQL persistence via SQLAlchemy. SQLite by default; MySQL optional (creds from .env).

Only *validated* data should be written here -- the data-quality gate runs first (pipeline.py).

CLI:
    python -m src.db.database --load
"""
from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import Engine, create_engine

from src.config import load_settings


def get_engine() -> Engine:
    """Return a SQLAlchemy engine from Settings.db_url (credentials never hardcoded)."""
    return create_engine(load_settings().db_url)


def write_df(df: pd.DataFrame, table: str, engine: Engine | None = None) -> int:
    """Write a DataFrame to `table` (replace) and return the row count.

    TODO: implement with df.to_sql(...); log rows written; ensure the target dir exists
    for SQLite. Keep it backend-agnostic so the same call works for SQLite and MySQL.
    """
    raise NotImplementedError


def read_df(table_or_query: str, engine: Engine | None = None) -> pd.DataFrame:
    """Read a table name or a SQL query into a DataFrame. TODO: implement."""
    raise NotImplementedError


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Load processed data into SQL.")
    parser.add_argument("--load", action="store_true")
    parser.parse_args()
    raise NotImplementedError("Wire to the processed dataset -- orchestrated by pipeline.py")


if __name__ == "__main__":
    _cli()
