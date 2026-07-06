"""Load configuration from config/config.yaml and secrets from the environment (.env).

No other module reads YAML or os.environ directly -- they import `load_settings()`. This is the fix
for the prototype's hardcoded DB password: credentials live in the environment, never in source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


@dataclass
class Settings:
    """Typed view over config.yaml plus a resolved SQLAlchemy DB URL."""

    raw: dict[str, Any]
    seed: int
    db_backend: str
    db_url: str

    def __getitem__(self, key: str) -> Any:
        """Convenience passthrough to the raw config dict (e.g. settings['quality'])."""
        return self.raw[key]


def _build_db_url(cfg: dict[str, Any]) -> str:
    """Return a SQLAlchemy URL from config + env. SQLite by default; Postgres if configured.

    Postgres credentials MUST come from environment variables -- never from config.yaml.
    """
    backend = os.environ.get("DB_BACKEND", cfg["database"]["backend"])
    if backend == "sqlite":
        path = os.environ.get("SQLITE_PATH", cfg["database"]["sqlite_path"])
        return f"sqlite:///{path}"
    if backend == "postgres":
        user = os.environ["POSTGRES_USER"]
        pwd = os.environ["POSTGRES_PASSWORD"]
        host = os.environ.get("POSTGRES_HOST", "db")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "cars")
        return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    raise ValueError(f"Unknown DB_BACKEND: {backend!r}")


def load_settings(config_path: Path = CONFIG_PATH) -> Settings:
    """Load .env, parse config.yaml, and return a Settings object.

    TODO: optionally validate that required keys/paths exist and raise a clear error early.
    """
    load_dotenv()
    cfg = yaml.safe_load(Path(config_path).read_text())
    return Settings(
        raw=cfg,
        seed=int(cfg.get("seed", 42)),
        db_backend=os.environ.get("DB_BACKEND", cfg["database"]["backend"]),
        db_url=_build_db_url(cfg),
    )
