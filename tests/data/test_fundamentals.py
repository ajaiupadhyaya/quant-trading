"""Tests for quant.data.fundamentals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant.data.fundamentals import (
    book_to_market,
    get_fundamentals,
    gross_profitability,
)


def test_get_fundamentals_returns_dict(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "marketCap": 3_000_000_000_000,
        "trailingPE": 28.4,
        "priceToBook": 45.0,
        "returnOnEquity": 1.5,
        "grossProfits": 175_000_000_000,
        "totalAssets": 350_000_000_000,
    }
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        info = get_fundamentals("AAPL")
    assert info["priceToBook"] == 45.0
    assert info["returnOnEquity"] == 1.5


def test_book_to_market_inverts_p_b(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {"priceToBook": 4.0}
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        assert book_to_market("AAPL") == pytest.approx(0.25)


def test_book_to_market_missing_returns_nan(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        result = book_to_market("AAPL")
    assert pd.isna(result)


def test_gross_profitability_divides_gp_by_assets(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {"grossProfits": 100.0, "totalAssets": 400.0}
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        assert gross_profitability("AAPL") == pytest.approx(0.25)
