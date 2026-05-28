"""US equity trading calendar utilities.

Light-weight NYSE calendar — knows about full closures (federal holidays + the
fixed market-specific dates: Good Friday, day-after-Thanksgiving early close)
and ``early_close_dates``. We deliberately avoid the heavier
``pandas-market-calendars`` dependency because the rules we need are stable
and well-known; embedding them keeps install footprint small and makes the
behavior reproducible without a third-party data fetch.

Coverage: 2010-01-01 through 2030-12-31. Each year is generated from a small
set of rules + the explicit one-off closures table.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the ``n``-th occurrence of ``weekday`` (Mon=0 … Sun=6) in (year, month)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of ``weekday`` in (year, month)."""
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    end = nxt - timedelta(days=1)
    offset = (end.weekday() - weekday) % 7
    return end - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm. Returns Easter Sunday for ``year``."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """Federal-holiday observation: Saturday -> previous Friday, Sunday -> next Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def _full_closures(year: int) -> frozenset[date]:
    """All full-day NYSE closures for ``year``."""
    closures: set[date] = set()

    # New Year's Day
    closures.add(_observed(date(year, 1, 1)))
    # MLK Day — third Monday of January (from 1998)
    closures.add(_nth_weekday(year, 1, 0, 3))
    # Presidents' Day — third Monday of February
    closures.add(_nth_weekday(year, 2, 0, 3))
    # Good Friday — Easter Sunday - 2
    closures.add(_easter_sunday(year) - timedelta(days=2))
    # Memorial Day — last Monday of May
    closures.add(_last_weekday(year, 5, 0))
    # Juneteenth — observed since 2022
    if year >= 2022:
        closures.add(_observed(date(year, 6, 19)))
    # Independence Day
    closures.add(_observed(date(year, 7, 4)))
    # Labor Day — first Monday of September
    closures.add(_nth_weekday(year, 9, 0, 1))
    # Thanksgiving — fourth Thursday of November
    closures.add(_nth_weekday(year, 11, 3, 4))
    # Christmas
    closures.add(_observed(date(year, 12, 25)))

    # One-off closures (national days of mourning, etc.). Add as needed.
    extras = {
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),  # Hurricane Sandy
        date(2018, 12, 5),  # George H.W. Bush state funeral
        date(2025, 1, 9),  # Jimmy Carter national day of mourning
    }
    for d in extras:
        if d.year == year:
            closures.add(d)

    return frozenset(closures)


@lru_cache(maxsize=64)
def _early_closes(year: int) -> frozenset[date]:
    """NYSE early-close days for ``year`` (13:00 ET).

    The fixed schedule is:
      * Day after Thanksgiving (Friday following 4th Thursday of November)
      * July 3 when July 4 is a weekday (or July 3 itself if a weekday)
      * Christmas Eve when Christmas is a weekday
    """
    early: set[date] = set()
    thx = _nth_weekday(year, 11, 3, 4)
    early.add(thx + timedelta(days=1))
    july4 = date(year, 7, 4)
    if july4.weekday() < 5:
        early.add(july4 - timedelta(days=1))
    xmas = date(year, 12, 25)
    if xmas.weekday() < 5:
        early.add(xmas - timedelta(days=1))
    return frozenset(early)


def is_trading_day(d: date) -> bool:
    """True iff ``d`` is a regular NYSE session (Mon-Fri, not a full-closure)."""
    if d.weekday() >= 5:
        return False
    return d not in _full_closures(d.year)


def is_early_close(d: date) -> bool:
    """True iff ``d`` is a trading day with a 13:00 ET early close."""
    return is_trading_day(d) and d in _early_closes(d.year)


def previous_trading_day(d: date) -> date:
    """Most recent trading day strictly before ``d``."""
    cur = d - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def next_trading_day(d: date) -> date:
    """First trading day strictly after ``d``."""
    cur = d + timedelta(days=1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur
