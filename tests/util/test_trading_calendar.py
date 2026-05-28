"""Tests for the embedded NYSE trading calendar."""

from __future__ import annotations

from datetime import date

import pytest

from quant.util.trading_calendar import (
    is_early_close,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)


@pytest.mark.parametrize(
    "day",
    [
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # MLK Day (3rd Monday Jan)
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday (Easter Mar 31)
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
        date(2025, 1, 1),
        date(2025, 1, 9),  # Jimmy Carter national day of mourning
        date(2025, 4, 18),  # Good Friday
        date(2026, 1, 19),  # MLK observed
    ],
)
def test_known_holidays_are_closed(day: date) -> None:
    assert not is_trading_day(day)


@pytest.mark.parametrize(
    "day",
    [
        date(2024, 1, 2),  # Tuesday after New Year's
        date(2024, 1, 16),  # Day after MLK
        date(2024, 7, 5),  # Friday after July 4
        date(2024, 11, 29),  # Day after Thanksgiving (early close, still open)
        date(2025, 7, 2),  # Wednesday before July 4
    ],
)
def test_known_trading_days_are_open(day: date) -> None:
    assert is_trading_day(day)


def test_weekends_are_closed() -> None:
    assert not is_trading_day(date(2024, 3, 2))  # Saturday
    assert not is_trading_day(date(2024, 3, 3))  # Sunday


def test_observed_rules_for_weekend_fixed_dates() -> None:
    # July 4, 2026 is a Saturday; observed on Friday July 3.
    assert not is_trading_day(date(2026, 7, 3))
    # July 4, 2027 is a Sunday; observed on Monday July 5.
    assert not is_trading_day(date(2027, 7, 5))


def test_early_close_days() -> None:
    # Day after Thanksgiving 2024
    assert is_early_close(date(2024, 11, 29))
    # July 3, 2025 (since July 4 is a Friday — early close)
    assert is_early_close(date(2025, 7, 3))


def test_previous_and_next_trading_day_skip_holidays() -> None:
    # Friday before MLK 2024 (Monday Jan 15) is Friday Jan 12
    assert previous_trading_day(date(2024, 1, 15)) == date(2024, 1, 12)
    # Next trading day after MLK is Tuesday Jan 16
    assert next_trading_day(date(2024, 1, 15)) == date(2024, 1, 16)


def test_hurricane_sandy_closures() -> None:
    assert not is_trading_day(date(2012, 10, 29))
    assert not is_trading_day(date(2012, 10, 30))
