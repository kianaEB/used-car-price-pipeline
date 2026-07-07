"""Optionally fetch a REAL used-car dataset (only if config dataset.source != 'synthetic').

`make data-real` runs this. The pipeline itself never requires it -- the default synthetic source
makes the whole project run offline with no accounts. Dispatches on config `dataset.source`:

  - url    : direct CSV link (implemented: streamed download + optional sha256 + cache)
  - kaggle : `kaggle datasets download` (needs the kaggle CLI + token) -- stub
  - openml : sklearn.datasets.fetch_openml (no account) -- stub
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from src.config import load_settings


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path, sha256: str | None = None) -> Path:
    """Stream a CSV from `url` to `dest` (cached; skip if present); verify an optional checksum."""
    import requests  # local import so the default synthetic path needs no network libs

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    if sha256:
        got = _sha256(dest)
        if got != sha256:
            raise ValueError(
                f"checksum mismatch for {dest}: expected {sha256}, got {got}"
            )
    return dest


def download_kaggle(dataset: str, dest_dir: Path) -> Path:
    """Fetch via the kaggle CLI (`kaggle datasets download -d <dataset> -p <dir> --unzip`).

    Needs the kaggle CLI on PATH + ~/.kaggle/kaggle.json. Returns the largest extracted CSV (the
    dataset itself, not a stray sample/license file).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(dest_dir),
            "--unzip",
        ],
        check=True,
    )
    csvs = list(dest_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSV extracted from {dataset!r} into {dest_dir}")
    return max(csvs, key=lambda p: p.stat().st_size)


def download_openml(name: str, dest: Path) -> Path:
    """Fetch via sklearn.datasets.fetch_openml(name=...) and write a CSV. TODO."""
    raise NotImplementedError


def main() -> int:
    """CLI entry for `make data-real`: fetch the configured source into paths.raw_file."""
    s = load_settings()
    ds = s["dataset"]
    source = ds.get("source", "synthetic")
    raw_file = Path(s["paths"]["raw_file"])

    if source == "synthetic":
        print(
            "dataset.source=synthetic -- nothing to download; `make all` already works offline."
        )
        return 0
    if source == "url":
        if not ds.get("url"):
            print("Set dataset.url in config/config.yaml for source=url.")
            return 1
        p = download_url(ds["url"], raw_file, ds.get("sha256") or None)
    elif source == "kaggle":
        p = download_kaggle(ds["kaggle_dataset"], raw_file.parent)
    elif source == "openml":
        p = download_openml(ds["openml_name"], raw_file)
    else:
        print(f"unknown dataset.source: {source}")
        return 1
    print(f"dataset ready -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
