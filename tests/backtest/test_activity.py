"""Tests for trade-activity metrics."""

from __future__ import annotations

import pandas as pd

from quant.backtest.activity import annualized_turnover


def _ledger(rows: list[tuple[int, float]]) -> pd.DataFrame:
    """Build a minimal trade ledger from (qty, fill_price) pairs."""
    return pd.DataFrame({"qty": [q for q, _ in rows], "fill_price": [p for _, p in rows]})


def _flat_equity(value: float, n_days: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.Series([value] * n_days, index=idx, name="equity")


def test_hand_computed_value():
    # two fills of $1000 notional each -> two-way $2000, one-way $1000.
    # mean equity $10,000 over exactly one trading year (252d).
    # (1000 / 10000) * (252 / 252) = 0.10
    trades = _ledger([(100, 10.0), (100, 10.0)])
    equity = _flat_equity(10_000.0, 252)
    assert annualized_turnover(trades, equity) == 0.10


def test_signed_qty_uses_absolute_notional():
    # qty sign (buy/sell direction) must not change turnover -- |qty| is used.
    # A +100 buy and a -100 sell @ $10 over 252d on $10,000 = same 0.10 as two buys.
    trades = _ledger([(100, 10.0), (-100, 10.0)])
    equity = _flat_equity(10_000.0, 252)
    assert annualized_turnover(trades, equity) == 0.10


def test_full_roundtrip_reads_as_100pct():
    # buy $10,000 then sell $10,000 on a $10,000 book over one year -> 1.0
    trades = _ledger([(1000, 10.0), (1000, 10.0)])
    equity = _flat_equity(10_000.0, 252)
    assert annualized_turnover(trades, equity) == 1.0


def test_annualization_scales_with_window_length():
    # identical trades over half a year -> double the one-year figure.
    trades = _ledger([(100, 10.0), (100, 10.0)])
    assert annualized_turnover(trades, _flat_equity(10_000.0, 126)) == 0.20


def test_homogeneous_in_scale():
    # scaling notional and equity by the same factor leaves turnover unchanged.
    base = annualized_turnover(_ledger([(100, 10.0), (100, 10.0)]), _flat_equity(10_000.0, 252))
    scaled = annualized_turnover(
        _ledger([(100, 100.0), (100, 100.0)]), _flat_equity(100_000.0, 252)
    )
    assert scaled == base


def test_empty_ledger_is_zero():
    assert (
        annualized_turnover(
            pd.DataFrame(columns=["qty", "fill_price"]), _flat_equity(10_000.0, 252)
        )
        == 0.0
    )


def test_empty_equity_is_zero():
    trades = _ledger([(100, 10.0)])
    assert annualized_turnover(trades, pd.Series(dtype=float)) == 0.0


def test_zero_mean_equity_is_zero():
    trades = _ledger([(100, 10.0)])
    assert annualized_turnover(trades, _flat_equity(0.0, 252)) == 0.0
