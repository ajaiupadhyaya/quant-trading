"""Tests for quant.data.universe."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant.data.universe import (
    ETF_UNIVERSE,
    etf_universe,
    load_sp500_snapshot,
    save_sp500_snapshot,
    sp500_constituents,
)


def test_etf_universe_is_fixed_eight() -> None:
    tickers = etf_universe()
    assert tickers == ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "EFA", "EEM"]
    assert tickers == ETF_UNIVERSE  # module-level constant matches


def test_save_and_load_sp500_round_trip(tmp_data_dir: Path) -> None:
    sample = ["AAPL", "MSFT", "GOOGL"]
    path = save_sp500_snapshot(sample, snapshot_date=date(2026, 5, 23), data_dir=tmp_data_dir)
    assert path.exists()
    assert path.parent == tmp_data_dir / "universe"

    loaded = load_sp500_snapshot(snapshot_date=date(2026, 5, 23), data_dir=tmp_data_dir)
    assert loaded == sample


def test_load_sp500_snapshot_missing_raises(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sp500_snapshot(snapshot_date=date(1999, 1, 1), data_dir=tmp_data_dir)


def test_sp500_constituents_from_wikipedia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the parser extracts tickers from a pd.read_html-style result."""
    fake_table = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "BRK.B", "GOOGL"],
            "Security": ["Apple", "Microsoft", "Berkshire", "Alphabet"],
        }
    )

    def fake_read_html(url: str, *args, **kwargs) -> list[pd.DataFrame]:
        assert "wikipedia.org" in url
        return [fake_table]

    monkeypatch.setattr("quant.data.universe.pd.read_html", fake_read_html)
    tickers = sp500_constituents()
    # Wikipedia uses "BRK.B" but Alpaca / yfinance use "BRK-B"
    assert "AAPL" in tickers
    assert "BRK-B" in tickers
    assert "BRK.B" not in tickers
