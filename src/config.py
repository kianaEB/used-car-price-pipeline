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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Top-level config sections the rest of the pipeline relies on; missing ones fail fast.
_REQUIRED_SECTIONS = ("paths", "database")


@dataclass
class Settings:
    """Typed view over config.yaml plus a resolved SQLAlchemy DB URL."""

    raw: dict[str, Any]
    seed: int
    db_backend: str
    db_url: str
    project_root: Path

    def __getitem__(self, key: str) -> Any:
        """Convenience passthrough to the raw config dict (e.g. settings['quality'])."""
        return self.raw[key]

    def path(self, name: str) -> Path:
        """Resolve a configured path (from the 'paths' block) to an absolute, root-anchored Path."""
        value = Path(self.raw["paths"][name])
        return value if value.is_absolute() else (self.project_root / value)


def _validate_config(cfg: dict[str, Any], config_path: Path) -> None:
    """Raise a clear ValueError if required sections/keys are missing from the parsed config."""
    missing = [section for section in _REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(
            f"Config at {config_path} is missing required section(s): {', '.join(missing)}"
        )
    if "backend" not in cfg["database"]:
        raise ValueError(f"Config at {config_path} is missing 'database.backend'")


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
    """Load .env, parse config.yaml, validate required sections, and return a Settings object."""
    load_dotenv()
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} did not parse to a mapping")
    _validate_config(cfg, config_path)
    return Settings(
        raw=cfg,
        seed=int(cfg.get("seed", 42)),
        db_backend=os.environ.get("DB_BACKEND", cfg["database"]["backend"]),
        db_url=_build_db_url(cfg),
        project_root=PROJECT_ROOT,
    )
