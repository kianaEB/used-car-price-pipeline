"""Tests for src.config: typed Settings, path resolution, and env-only DB URL logic.

These pin the two things that matter for stage 1: the SQLite/Postgres URL rules from SPEC section 10
(credentials from the environment, never config.yaml) and fail-fast validation of a bad config.
`load_dotenv` is neutralised so the suite never depends on a developer's local .env.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from src import config
from src.config import CONFIG_PATH, load_settings

PG_ENV = {
    "DB_BACKEND": "postgres",
    "POSTGRES_USER": "cars",
    "POSTGRES_PASSWORD": "s3cret",
    "POSTGRES_HOST": "db",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "cars",
}


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make config tests hermetic: never let a real .env leak DB env vars into a test."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)


@contextmanager
def _temp_config(body: str) -> Iterator[Path]:
    """Yield the path to a throwaway config.yaml holding `body` (stdlib tempfile; auto-cleaned)."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(body)
        yield path


def test_defaults_use_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped config resolves to SQLite with the configured seed and no external creds."""
    for key in PG_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SQLITE_PATH", raising=False)
    settings = load_settings()
    assert settings.seed == 42
    assert settings.db_backend == "sqlite"
    assert settings.db_url.startswith("sqlite:///")


def test_committed_config_defaults_to_offline_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env overrides, the source stays synthetic so CI / make all run offline."""
    for key in ("DATASET_SOURCE", "FRESHNESS_REFERENCE_DATE", "BATCHING_FREQ"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings()
    assert settings["dataset"]["source"] == "synthetic"


def test_env_overrides_wire_a_real_dataset_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run opts in via env (source, reference date, freq, error ceiling) without config edits."""
    monkeypatch.setenv("DATASET_SOURCE", "kaggle")
    monkeypatch.setenv("FRESHNESS_REFERENCE_DATE", "2021-05-04")
    monkeypatch.setenv("BATCHING_FREQ", "D")
    monkeypatch.setenv("QUALITY_MAX_ERROR_FRACTION", "0.6")
    settings = load_settings()
    assert settings["dataset"]["source"] == "kaggle"
    assert settings["monitoring"]["freshness"]["reference_date"] == "2021-05-04"
    assert settings["batching"]["freq"] == "D"
    assert settings["quality"]["max_error_fraction"] == 0.6  # cast from string to float


def test_getitem_passthrough() -> None:
    """Settings indexes straight into the raw config for nested sections."""
    settings = load_settings()
    assert settings["database"]["backend"] == "sqlite"
    assert settings["quality"]["min_rows"] == 100


def test_path_resolves_absolute_under_root() -> None:
    """path() turns a relative 'paths' entry into an absolute path anchored at the project root."""
    settings = load_settings()
    processed = settings.path("processed_dir")
    assert processed.is_absolute()
    assert processed == settings.project_root / "data" / "processed"
    assert settings.project_root in processed.parents


def test_postgres_url_built_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Postgres backend assembles its URL entirely from environment variables."""
    for key, value in PG_ENV.items():
        monkeypatch.setenv(key, value)
    settings = load_settings()
    assert settings.db_backend == "postgres"
    assert settings.db_url == "postgresql+psycopg2://cars:s3cret@db:5432/cars"


def test_missing_postgres_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A postgres backend with no password in the environment fails loudly (no silent default)."""
    for key, value in PG_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(KeyError):
        load_settings()


def test_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognised backend is rejected rather than silently falling through."""
    monkeypatch.setenv("DB_BACKEND", "mysql")
    with pytest.raises(ValueError, match="Unknown DB_BACKEND"):
        load_settings()


def test_missing_config_file_raises() -> None:
    """A nonexistent config path fails with a clear FileNotFoundError, not a downstream KeyError."""
    with pytest.raises(FileNotFoundError):
        load_settings(Path("does") / "not" / "exist.yaml")


def test_missing_required_section_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config missing a required section (e.g. 'database') raises a clear ValueError."""
    monkeypatch.delenv("DB_BACKEND", raising=False)
    with _temp_config("paths:\n  raw_dir: data/raw\nseed: 7\n") as path:
        with pytest.raises(ValueError, match="database"):
            load_settings(path)


def test_missing_database_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 'database' section without a 'backend' key raises a clear ValueError."""
    monkeypatch.delenv("DB_BACKEND", raising=False)
    body = "paths:\n  raw_dir: data/raw\ndatabase:\n  sqlite_path: x.db\n"
    with _temp_config(body) as path:
        with pytest.raises(ValueError, match="database.backend"):
            load_settings(path)


def test_non_mapping_config_raises() -> None:
    """A config file that does not parse to a mapping raises a clear ValueError."""
    with _temp_config("just a bare scalar\n") as path:
        with pytest.raises(ValueError, match="did not parse to a mapping"):
            load_settings(path)


def test_shipped_config_has_no_hardcoded_credentials() -> None:
    """Golden rule 2: the checked-in config declares no DB user/password (creds are env-only)."""
    database_section = load_settings()["database"]
    assert "password" not in database_section
    assert "user" not in database_section
    assert "password" not in CONFIG_PATH.read_text().lower()
