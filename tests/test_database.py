"""Tests for SQL write/read round-trip and the engine URL logic (no live Postgres needed).

Uses a temp-file SQLite engine (disposed before cleanup) to dodge the machine's locked pytest
tmp_path base. The Postgres case only inspects the built URL -- it never opens a connection.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine

from src import config
from src.db.database import _cli, get_engine, read_df, write_df

PG_ENV = {
    "DB_BACKEND": "postgres",
    "POSTGRES_USER": "cars",
    "POSTGRES_PASSWORD": "s3cret",
    "POSTGRES_HOST": "db",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "cars",
}


@contextmanager
def _engine() -> Iterator[Engine]:
    """A throwaway SQLite engine on a temp file, disposed before the dir is cleaned up."""
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{(Path(directory) / 'cars.db').as_posix()}")
        try:
            yield engine
        finally:
            engine.dispose()


# --------------------------------------------------------------------------- write/read round-trip


def test_write_read_roundtrip_preserves_values_and_dtypes(clean_df):
    with _engine() as engine:
        n = write_df(clean_df, "cars", engine)
        assert n == len(clean_df)
        out = read_df("cars", engine)
    pd.testing.assert_frame_equal(out, clean_df)  # values AND dtypes preserved
    assert pd.api.types.is_integer_dtype(out["year"])
    assert pd.api.types.is_float_dtype(out["price"])
    assert pd.api.types.is_string_dtype(out["brand"])


def test_write_df_replace_is_idempotent(clean_df):
    with _engine() as engine:
        write_df(clean_df, "cars", engine)
        write_df(clean_df, "cars", engine)  # replace, not append
        assert len(read_df("cars", engine)) == len(clean_df)


def test_write_df_append_accumulates(clean_df):
    with _engine() as engine:
        write_df(clean_df, "cars", engine, if_exists="append")
        write_df(clean_df, "cars", engine, if_exists="append")
        assert len(read_df("cars", engine)) == 2 * len(clean_df)


def test_read_df_accepts_a_sql_query(clean_df):
    with _engine() as engine:
        write_df(clean_df, "cars", engine)
        out = read_df("SELECT * FROM cars WHERE price > 10000", engine)
    assert (out["price"] > 10000).all()
    assert len(out) == int((clean_df["price"] > 10000).sum())


# --------------------------------------------------------------------------------- get_engine URL


@pytest.fixture
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make engine tests hermetic: never let a local .env leak DB vars into the config."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)


def test_get_engine_defaults_to_sqlite(_no_dotenv, monkeypatch: pytest.MonkeyPatch):
    for key in PG_ENV:
        monkeypatch.delenv(key, raising=False)
    engine = get_engine()
    try:
        assert engine.url.drivername == "sqlite"
        assert engine.url.database.endswith("cars.db")
    finally:
        engine.dispose()


def test_get_engine_builds_postgres_url_from_env(
    _no_dotenv, monkeypatch: pytest.MonkeyPatch
):
    for key, value in PG_ENV.items():
        monkeypatch.setenv(key, value)
    engine = get_engine()  # no connection is opened
    try:
        assert engine.url.drivername == "postgresql+psycopg2"
        assert engine.url.username == "cars"
        assert engine.url.host == "db"
        assert engine.url.database == "cars"
    finally:
        engine.dispose()


def test_cli_without_load_flag_prints_help(monkeypatch: pytest.MonkeyPatch, capsys):
    """`python -m src.db.database` with no --load prints help and writes nothing (exit 0)."""
    monkeypatch.setattr(sys, "argv", ["database"])
    assert _cli() == 0
    assert "usage" in capsys.readouterr().out.lower()
