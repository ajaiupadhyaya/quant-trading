"""Data quality checks for daily OHLCV bar caches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class SymbolQuality:
    symbol: str
    rows: int
    missing_bars: int
    duplicate_timestamps: int
    impossible_ohlc: int
    stale: bool

    @property
    def passed(self) -> bool:
        return (
            self.missing_bars == 0
            and self.duplicate_timestamps == 0
            and self.impossible_ohlc == 0
            and not self.stale
        )


@dataclass(frozen=True)
class DataQualityReport:
    start: date
    end: date
    symbols: dict[str, SymbolQuality]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.symbols.values())


def evaluate_bar_quality(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    stale_after_days: int = 7,
) -> DataQualityReport:
    expected = pd.bdate_range(start, end)
    out: dict[str, SymbolQuality] = {}
    for symbol, bars in sorted(bars_by_symbol.items()):
        if bars.empty:
            out[symbol] = SymbolQuality(symbol, 0, len(expected), 0, 0, True)
            continue
        idx = pd.DatetimeIndex(bars.index).normalize()
        unique_idx = pd.DatetimeIndex(idx.unique()).sort_values()
        duplicate_count = int(idx.duplicated().sum())
        missing = len(expected.difference(unique_idx))
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        open_ = bars["open"].astype(float)
        close = bars["close"].astype(float)
        impossible = int(((high < low) | (open_ > high) | (open_ < low) | (close > high) | (close < low)).sum())
        last_date = unique_idx.max().date()
        stale = (pd.Timestamp(end).date() - last_date).days > stale_after_days
        out[symbol] = SymbolQuality(
            symbol=symbol,
            rows=len(bars),
            missing_bars=missing,
            duplicate_timestamps=duplicate_count,
            impossible_ohlc=impossible,
            stale=stale,
        )
    return DataQualityReport(start=start, end=end, symbols=out)
