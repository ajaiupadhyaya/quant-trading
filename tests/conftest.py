"""Shared pytest fixtures and configuration."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Provide an isolated data directory and point QUANT_DATA_DIR at it."""
    data = tmp_path / "data"
    for sub in ("universe", "raw", "backtests", "live", "features", "fundamentals", "macro"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANT_DATA_DIR", str(data))
    yield data


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate env with dummy credentials so Settings() doesn't fail."""
    monkeypatch.setenv("ALPACA_API_KEY", "PKTEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SECRETTEST")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("FRED_API_KEY", "FREDTEST")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
