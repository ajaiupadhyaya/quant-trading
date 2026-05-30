# quant/intraday/data/adjustments.py
"""Point-in-time corporate-action adjustment. Raw prices are never rewritten;
splits/dividends are applied at READ time, capped by an `as_of` date so a
backtest only ever sees actions known by then (charter principle #1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

_PRICE_COLUMNS = ("open", "high", "low", "close", "bid", "ask", "price", "vwap")


@dataclass(frozen=True)
class Adjustment:
    """A corporate action. `cash_dividend` is the POST-split dollar amount per
    share (subtracted from pre-ex prices after the split ratio is applied)."""

    ex_date: date
    split_ratio: float  # e.g. 4.0 means 4-for-1; price divided by 4 before ex-date
    cash_dividend: float  # post-split $/share, subtracted from pre-ex prices


def adjust_prices(df: pd.DataFrame, factors: list[Adjustment], as_of: date) -> pd.DataFrame:
    """Back-adjust price columns for splits/dividends with ex_date <= as_of.

    Bars on/after an ex-date are the "current" scale; bars strictly before it
    are divided by the split ratio (and reduced by the dividend) so the series
    is continuous. Actions with ex_date > as_of are ignored (no lookahead).
    """
    applicable = [f for f in factors if f.ex_date <= as_of]
    if not applicable:
        return df.copy()
    out = df.copy()
    price_cols = [c for c in out.columns if c in _PRICE_COLUMNS]
    dt_index = pd.DatetimeIndex(out.index)
    tz = dt_index.tz
    ex_index = dt_index.tz_convert("UTC") if tz is not None else dt_index
    for adj in applicable:
        # Match the index's tz-awareness so the comparison is homogeneous
        # (a tz-aware ex_ts vs a tz-naive index raises InvalidComparison).
        ex_ts = pd.Timestamp(adj.ex_date, tz="UTC") if tz is not None else pd.Timestamp(adj.ex_date)
        pre = ex_index < ex_ts
        if adj.split_ratio and adj.split_ratio != 1.0:
            out.loc[pre, price_cols] = out.loc[pre, price_cols] / adj.split_ratio
        if adj.cash_dividend:
            out.loc[pre, price_cols] = out.loc[pre, price_cols] - adj.cash_dividend
    return out
