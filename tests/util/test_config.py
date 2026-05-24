"""Tests for quant.util.config.Settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant.util.config import Settings


def test_settings_reads_required_env(fake_env: None) -> None:
    settings = Settings()
    assert settings.alpaca_api_key == "PKTEST"
    assert settings.alpaca_secret_key == "SECRETTEST"
    assert settings.fred_api_key == "FREDTEST"


def test_settings_paper_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setenv("FRED_API_KEY", "z")
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    settings = Settings()
    assert settings.alpaca_paper is True


def test_settings_data_dir_resolves(fake_env: None, tmp_data_dir: Path) -> None:
    settings = Settings()
    assert settings.data_dir == tmp_data_dir
    assert settings.data_dir.is_dir()


def test_settings_paper_url_for_paper(fake_env: None) -> None:
    settings = Settings()
    assert "paper-api.alpaca.markets" in settings.alpaca_base_url
