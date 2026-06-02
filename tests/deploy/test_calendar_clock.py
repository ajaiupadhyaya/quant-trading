"""Pure ET clock + session-close helpers (DST + early-close)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

from quant.deploy.calendar_clock import (
    EARLY_CLOSE,
    NORMAL_CLOSE,
    is_trading_day,
    session_close_et,
    to_et,
)


def test_to_et_winter_is_utc_minus_5() -> None:
    # 2026-01-15 20:00 UTC -> 15:00 EST
    et = to_et(datetime(2026, 1, 15, 20, 0, tzinfo=UTC))
    assert (et.hour, et.minute) == (15, 0)


def test_to_et_summer_is_utc_minus_4() -> None:
    # 2026-07-15 19:55 UTC -> 15:55 EDT
    et = to_et(datetime(2026, 7, 15, 19, 55, tzinfo=UTC))
    assert (et.hour, et.minute) == (15, 55)


def test_to_et_accepts_naive_as_utc() -> None:
    et = to_et(datetime(2026, 7, 15, 19, 55))
    assert (et.hour, et.minute) == (15, 55)


def test_session_close_normal_day() -> None:
    assert session_close_et(date(2026, 6, 2)) == NORMAL_CLOSE == time(16, 0)


def test_session_close_early_close_day_after_thanksgiving() -> None:
    # 2026 Thanksgiving = 4th Thu Nov = Nov 26; day after = Nov 27 (early close).
    assert session_close_et(date(2026, 11, 27)) == EARLY_CLOSE == time(13, 0)


def test_is_trading_day_reexport() -> None:
    assert is_trading_day(date(2026, 6, 2)) is True
    assert is_trading_day(date(2026, 6, 6)) is False  # Saturday
