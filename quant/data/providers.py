"""Provider boundary for swapping daily bar data sources."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class BarProvider(Protocol):
    """Any paid/free bar provider that can return strategy-compatible daily bars."""

    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return a wide DataFrame with MultiIndex columns `(symbol, field)`."""


def get_provider_bars(
    provider: BarProvider,
    symbols: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    bars = provider.get_daily_bars(symbols, start, end)
    if not isinstance(bars.columns, pd.MultiIndex):
        raise ValueError("bar provider must return MultiIndex columns")
    return bars.sort_index()
