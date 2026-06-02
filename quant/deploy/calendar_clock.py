"""Pure ET wall-clock + session-close classification for the tick scheduler.

UTC has no DST, so converting an aware UTC instant to America/New_York is always
unambiguous — this is what makes the scheduler DST-correct (the old GitHub crons
were fixed-UTC and drifted +1h in winter). Trading-day / early-close facts are
delegated to quant/util/trading_calendar.py (single source of truth). No I/O and
no datetime.now() here — the current time is always an argument.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from quant.util.trading_calendar import is_early_close, is_trading_day

ET = ZoneInfo("America/New_York")
NORMAL_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

__all__ = ["EARLY_CLOSE", "ET", "NORMAL_CLOSE", "is_trading_day", "session_close_et", "to_et"]


def to_et(now_utc: datetime) -> datetime:
    """Convert an instant to America/New_York. A naive datetime is assumed UTC."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    return now_utc.astimezone(ET)


def session_close_et(d: date) -> time:
    """NYSE close time (ET) for trading day ``d``: 13:00 on early-close days, else 16:00.

    Raises ValueError on a non-trading day (callers gate on the day rule first).
    """
    if not is_trading_day(d):
        raise ValueError(f"{d} is not a trading day")
    return EARLY_CLOSE if is_early_close(d) else NORMAL_CLOSE
