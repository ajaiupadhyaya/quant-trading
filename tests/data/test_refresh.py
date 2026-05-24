"""Tests for quant.data.refresh."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.data.refresh import refresh_caches


def test_refresh_calls_get_bars_with_union_of_universes(
    tmp_data_dir: Path,
    fake_env: None,
) -> None:
    fake_aapl = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="timestamp"),
    )
    fake_spy = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="timestamp"),
    )

    def fake_fetch(symbols, start, end, settings):
        return {sym: fake_aapl if sym != "SPY" else fake_spy for sym in symbols}

    with (
        patch("quant.data.bars._fetch_alpaca", side_effect=fake_fetch),
        patch("quant.data.universe.sp500_constituents", return_value=["AAPL", "MSFT"]),
    ):
        report = refresh_caches(start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert report.symbols_fetched >= 8 + 2  # 8 ETFs + at least AAPL,MSFT
    # ETF universe always included:
    for etf in ("SPY", "TLT", "GLD"):
        assert etf in report.symbols
