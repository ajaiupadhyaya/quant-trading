# quant/intraday/data/quality.py
"""Data-quality guards: session calendar, gap detection, bad-tick filtering, doctor."""

from __future__ import annotations

from datetime import date

import pandas as pd


def regular_session_minutes(day: date) -> int:
    """Number of 1-minute bars in a regular US equity session (9:30–16:00 ET).

    Half-days return 210; this base implementation returns 390 for any weekday.
    (A half-day calendar can be layered in later from the market calendar.)
    """
    return 390


def detect_minute_gaps(bars: pd.DataFrame) -> list[pd.Timestamp]:
    """Return the missing minute timestamps between the first and last bar."""
    if bars.empty:
        return []
    full = pd.date_range(bars.index.min(), bars.index.max(), freq="1min")
    return [ts for ts in full if ts not in bars.index]


def filter_bad_trades(trades: pd.DataFrame, ref_price: float, max_deviation: float = 0.2) -> pd.DataFrame:
    """Drop non-positive prices and prints more than `max_deviation` from ref_price."""
    if trades.empty:
        return trades
    lo, hi = ref_price * (1 - max_deviation), ref_price * (1 + max_deviation)
    mask = (trades["price"] > 0) & (trades["price"] >= lo) & (trades["price"] <= hi)
    return trades.loc[mask]


def run_doctor(data_root: object) -> dict[str, int]:
    """Summarize the intraday store: partition counts per dataset."""
    from pathlib import Path

    counts: dict[str, int] = {}
    for ds in ("trades", "quote_bars_1s", "minute_bars"):
        d = Path(data_root) / ds
        counts[ds] = len(list(d.rglob("*.parquet"))) if d.exists() else 0
    return counts
