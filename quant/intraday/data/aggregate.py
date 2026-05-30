# quant/intraday/data/aggregate.py
"""Pure aggregation: raw ticks -> bars. No I/O, fully unit-testable."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MINUTE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]


def trades_to_minute_bars(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Resample a (DatetimeIndex, price, size) trade frame to 1-minute OHLCV.

    `symbol` is accepted for API symmetry/logging; output is single-symbol.
    """
    if trades.empty:
        return pd.DataFrame(columns=_MINUTE_COLUMNS)
    df = trades.sort_index()
    grouped = df.resample("1min", label="left", closed="left")
    notional = (df["price"] * df["size"]).resample("1min", label="left", closed="left").sum()
    volume = grouped["size"].sum()
    out = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": volume.astype("int64"),
            "vwap": np.where(volume > 0, notional / volume.replace(0, np.nan), np.nan),
            "trade_count": grouped["price"].count().astype("int64"),
        }
    )
    return out.dropna(subset=["open"])[_MINUTE_COLUMNS]
