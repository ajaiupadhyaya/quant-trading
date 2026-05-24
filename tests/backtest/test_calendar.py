"""Tests for quant.backtest.calendar.is_rebalance_day."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.backtest.calendar import is_rebalance_day


@pytest.fixture
def history_jan2024() -> pd.DatetimeIndex:
    """Business days in January 2024 — 22 bars."""
    return pd.bdate_range("2024-01-01", "2024-01-31")


def test_daily_always_rebalances(history_jan2024: pd.DatetimeIndex) -> None:
    for ts in history_jan2024:
        assert is_rebalance_day(ts.date(), "daily", history_jan2024) is True


def test_weekly_only_first_business_day_of_iso_week(history_jan2024: pd.DatetimeIndex) -> None:
    # Jan 2024 ISO weeks: w1 (Mon Jan 1), w2 (Mon Jan 8), w3 (Mon Jan 15),
    # w4 (Mon Jan 22), w5 (Mon Jan 29). Each Monday is a business day in Jan 2024.
    expected_mondays = {
        date(2024, 1, 1),
        date(2024, 1, 8),
        date(2024, 1, 15),
        date(2024, 1, 22),
        date(2024, 1, 29),
    }
    got = {
        ts.date()
        for ts in history_jan2024
        if is_rebalance_day(ts.date(), "weekly", history_jan2024)
    }
    assert got == expected_mondays


def test_weekly_handles_holiday_kick(history_jan2024: pd.DatetimeIndex) -> None:
    """If the ISO Monday is a holiday and missing from history, the next bar is the rebalance."""
    # Drop Jan 1, 2024 (assume holiday): first bar of that ISO week becomes Tue Jan 2.
    history = history_jan2024.drop(pd.Timestamp("2024-01-01"))
    got = [ts.date() for ts in history if is_rebalance_day(ts.date(), "weekly", history)]
    assert date(2024, 1, 2) in got
    assert date(2024, 1, 1) not in got


def test_monthly_only_first_business_day_of_month() -> None:
    history = pd.bdate_range("2024-01-01", "2024-03-31")
    rebalances = [ts.date() for ts in history if is_rebalance_day(ts.date(), "monthly", history)]
    assert rebalances == [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]


def test_invalid_frequency_raises(history_jan2024: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="frequency"):
        is_rebalance_day(date(2024, 1, 2), "annually", history_jan2024)
