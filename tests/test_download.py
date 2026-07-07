"""Tests for the optional real-dataset fetchers (no network: the kaggle CLI is mocked)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.ingest import download


def test_download_kaggle_shells_out_and_returns_the_extracted_csv(monkeypatch):
    """download_kaggle runs the documented kaggle command and returns the largest extracted CSV."""
    captured = {}

    with tempfile.TemporaryDirectory() as directory:
        dest = Path(directory)

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["check"] = kwargs.get("check")
            (dest / "license.csv").write_text("a\n")  # a small stray file
            (dest / "vehicles.csv").write_text(
                "price,manufacturer\n1,ford\n"
            )  # the dataset
            return None

        monkeypatch.setattr(download.subprocess, "run", fake_run)
        result = download.download_kaggle("owner/name", dest)

    assert result.name == "vehicles.csv"  # the largest CSV, not the stray license file
    assert captured["cmd"] == [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "owner/name",
        "-p",
        str(dest),
        "--unzip",
    ]
    assert captured["check"] is True


def test_download_kaggle_raises_when_nothing_extracted(monkeypatch):
    """A clear error when the CLI succeeds but no CSV lands (bad dataset id / layout)."""
    with tempfile.TemporaryDirectory() as directory:
        dest = Path(directory)
        monkeypatch.setattr(download.subprocess, "run", lambda cmd, **k: None)
        with pytest.raises(FileNotFoundError):
            download.download_kaggle("owner/name", dest)
