"""Tests for the pure financing-charge model."""

from __future__ import annotations

import pytest

from quant.backtest.financing import financing_charge

_BORROW = 50.0  # bps/yr
_FIN = 200.0  # bps/yr


def test_single_short_one_day():
    # short 100 @ $50 = $5,000 notional; 50 bps/yr; 1 day; positive cash.
    c = financing_charge(
        {"AAA": -100},
        {"AAA": 50.0},
        cash=10_000.0,
        days_elapsed=1,
        annual_borrow_bps=_BORROW,
        annual_financing_bps=_FIN,
    )
    assert c.borrow_cost == pytest.approx(5_000.0 * (50.0 / 1e4) * (1 / 365))
    assert c.margin_financing_cost == 0.0
    assert c.total == pytest.approx(c.borrow_cost)


def test_long_only_positive_cash_is_zero():
    c = financing_charge(
        {"AAA": 100},
        {"AAA": 50.0},
        cash=10_000.0,
        days_elapsed=1,
        annual_borrow_bps=_BORROW,
        annual_financing_bps=_FIN,
    )
    assert c.borrow_cost == 0.0
    assert c.margin_financing_cost == 0.0


def test_weekend_gap_is_three_days():
    one = financing_charge({"AAA": -100}, {"AAA": 50.0}, 10_000.0, 1, _BORROW, _FIN)
    three = financing_charge({"AAA": -100}, {"AAA": 50.0}, 10_000.0, 3, _BORROW, _FIN)
    assert three.borrow_cost == pytest.approx(3.0 * one.borrow_cost)


def test_margin_debit_is_financed():
    # negative cash -2,000; 200 bps/yr; 1 day; no shorts.
    c = financing_charge(
        {"AAA": 100},
        {"AAA": 50.0},
        cash=-2_000.0,
        days_elapsed=1,
        annual_borrow_bps=_BORROW,
        annual_financing_bps=_FIN,
    )
    assert c.borrow_cost == 0.0
    assert c.margin_financing_cost == pytest.approx(2_000.0 * (200.0 / 1e4) * (1 / 365))


def test_zero_or_negative_days_is_zero():
    for d in (0, -5):
        c = financing_charge({"AAA": -100}, {"AAA": 50.0}, -2_000.0, d, _BORROW, _FIN)
        assert c.total == 0.0


def test_combined_short_and_debit():
    c = financing_charge(
        {"AAA": -100},
        {"AAA": 50.0},
        cash=-2_000.0,
        days_elapsed=1,
        annual_borrow_bps=_BORROW,
        annual_financing_bps=_FIN,
    )
    exp_borrow = 5_000.0 * (50.0 / 1e4) * (1 / 365)
    exp_fin = 2_000.0 * (200.0 / 1e4) * (1 / 365)
    assert c.borrow_cost == pytest.approx(exp_borrow)
    assert c.margin_financing_cost == pytest.approx(exp_fin)
    assert c.total == pytest.approx(exp_borrow + exp_fin)


def test_missing_or_nonfinite_prior_close_contributes_zero():
    # one short missing from prior_close, one with NaN price -> both contribute 0.
    c = financing_charge(
        {"AAA": -100, "BBB": -100},
        {"BBB": float("nan")},
        cash=10_000.0,
        days_elapsed=1,
        annual_borrow_bps=_BORROW,
        annual_financing_bps=_FIN,
    )
    assert c.borrow_cost == 0.0
