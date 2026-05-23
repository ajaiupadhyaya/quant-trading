"""Tests for quant.data.bars."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.data.bars import BarRequest, _cache_path, _read_cache, _write_cache, get_bars


def _fake_alpaca_frame(symbol: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [1_000_000 + i for i in range(len(dates))],
        },
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="timestamp"),
    )


def test_cache_round_trip(tmp_data_dir: Path) -> None:
    df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    path = _cache_path("AAPL", tmp_data_dir)
    _write_cache(df, path)
    assert path.exists()
    loaded = _read_cache(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_get_bars_cache_hit(tmp_data_dir: Path, fake_env: None) -> None:
    df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    _write_cache(df, _cache_path("AAPL", tmp_data_dir))

    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    with patch("quant.data.bars._fetch_alpaca") as mock_alpaca:
        result = get_bars(req)
    mock_alpaca.assert_not_called()
    assert ("AAPL", "close") in result.columns
    assert len(result) == 2


def test_get_bars_cache_miss_calls_alpaca(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    with patch("quant.data.bars._fetch_alpaca", return_value={"AAPL": fake_df}) as mock_alpaca:
        result = get_bars(req)
    mock_alpaca.assert_called_once()
    assert ("AAPL", "close") in result.columns
    assert _cache_path("AAPL", tmp_data_dir).exists()


def test_get_bars_alpaca_failure_falls_back_to_yfinance(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    with (
        patch("quant.data.bars._fetch_alpaca", side_effect=RuntimeError("alpaca down")),
        patch("quant.data.bars._fetch_yfinance", return_value={"AAPL": fake_df}) as mock_yf,
    ):
        result = get_bars(req)
    mock_yf.assert_called_once()
    assert ("AAPL", "close") in result.columns


def test_get_bars_multi_symbol_result_shape(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL", "MSFT"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_aapl = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    fake_msft = _fake_alpaca_frame("MSFT", [date(2024, 1, 2), date(2024, 1, 3)])
    with patch(
        "quant.data.bars._fetch_alpaca",
        return_value={"AAPL": fake_aapl, "MSFT": fake_msft},
    ):
        result = get_bars(req)
    # Multi-index columns: (symbol, field)
    assert set(result.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert "close" in result.columns.get_level_values(1)
