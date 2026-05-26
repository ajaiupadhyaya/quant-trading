"""Tests for quant/util/calendar.py — the prior_trading_day shim."""

from __future__ import annotations

from datetime import date

from quant.util.calendar import prior_trading_day


def test_prior_trading_day_skips_memorial_day_2026() -> None:
    # Memorial Day 2026 is Monday May 25. Prior trading day from Tue 5/26 is Fri 5/22.
    assert prior_trading_day(date(2026, 5, 26)) == date(2026, 5, 22)
