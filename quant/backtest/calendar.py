"""Predicate: should the strategy rebalance on this bar?"""

from __future__ import annotations

from datetime import date

import pandas as pd

_VALID_FREQUENCIES = {"daily", "weekly", "monthly"}


def is_rebalance_day(
    asof: date,
    frequency: str,
    history: pd.DatetimeIndex,
) -> bool:
    """Return True if `asof` is a rebalance day for `frequency` given `history`.

    `history` is the full index of bars the engine is iterating over. For weekly
    and monthly frequencies, the first bar of each ISO week / calendar month in
    `history` is the rebalance day — this correctly handles holiday-shifted
    Mondays / month-starts without hard-coding a calendar.
    """
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(
            f"Unknown frequency {frequency!r}; expected one of {sorted(_VALID_FREQUENCIES)}"
        )
    ts = pd.Timestamp(asof)
    if ts not in history:
        return False
    if frequency == "daily":
        return True

    if frequency == "weekly":
        key = (ts.isocalendar().year, ts.isocalendar().week)
        same_week = [t for t in history if (t.isocalendar().year, t.isocalendar().week) == key]
        return bool(same_week and same_week[0] == ts)

    # monthly
    same_month = [t for t in history if t.year == ts.year and t.month == ts.month]
    return bool(same_month and same_month[0] == ts)
