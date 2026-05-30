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


_QUOTE_COLUMNS = ["bid", "ask", "bid_size", "ask_size"]


def quotes_to_second_bars(quotes: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Downsample raw NBBO quotes to 1-second bars (last quote in each second).

    Sampling the *last* quote per second gives the prevailing NBBO at second
    close — the spread a marketable order would face at that instant.
    """
    if quotes.empty:
        return pd.DataFrame(columns=_QUOTE_COLUMNS)
    df = quotes.sort_index()
    out = df.resample("1s", label="left", closed="left").last().dropna(subset=["bid", "ask"])
    # NBBO size can be NaN during auctions/halts/outages even with a valid bid/ask;
    # keep the spread, treat missing size as 0 rather than crashing the int cast.
    out["bid_size"] = out["bid_size"].fillna(0).astype("int64")
    out["ask_size"] = out["ask_size"].fillna(0).astype("int64")
    return out[_QUOTE_COLUMNS]
