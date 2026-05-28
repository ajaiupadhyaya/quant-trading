"""Tests for paid-data-ready provider boundary."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant.data.providers import BarProvider, get_provider_bars


class _Provider:
    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        idx = pd.DatetimeIndex([pd.Timestamp(start)], name="timestamp")
        return pd.concat(
            {
                symbol: pd.DataFrame(
                    {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
                    index=idx,
                )
                for symbol in symbols
            },
            axis=1,
        )


def test_bar_provider_boundary_returns_strategy_compatible_frame() -> None:
    provider: BarProvider = _Provider()
    bars = get_provider_bars(provider, ["SPY"], date(2026, 1, 2), date(2026, 1, 2))

    assert ("SPY", "close") in bars.columns
    assert len(bars) == 1
