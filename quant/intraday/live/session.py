"""Equities-session resolver for the intraday loop. RTH 09:30-16:00 ET, with NYSE
early closes (13:00 ET) honored via the existing trading calendar."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from quant.util import trading_calendar as cal

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)


@dataclass(frozen=True)
class SessionState:
    open: bool
    close: datetime  # today's RTH close (tz-aware, in `now`'s tz)


def session_state(now: datetime) -> SessionState:
    et = now.astimezone(_ET)
    close_time = _EARLY_CLOSE if cal.is_early_close(et.date()) else _CLOSE
    close_et = datetime.combine(et.date(), close_time, tzinfo=_ET)
    is_open = cal.is_trading_day(et.date()) and (_OPEN <= et.time() < close_time)
    return SessionState(open=is_open, close=close_et.astimezone(now.tzinfo or _ET))
