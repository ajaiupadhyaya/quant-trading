"""Tests for trade-activity metrics."""

from __future__ import annotations

import math

import pandas as pd

from quant.backtest.activity import annualized_turnover, capacity_report


def _ledger(rows: list[tuple[int, float]]) -> pd.DataFrame:
    """Build a minimal trade ledger from (qty, fill_price) pairs."""
    return pd.DataFrame({"qty": [q for q, _ in rows], "fill_price": [p for _, p in rows]})


def _ledger_adv(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """Trade ledger from (qty, fill_price, adv_dollar) triples."""
    return pd.DataFrame(
        {
            "qty": [q for q, _, _ in rows],
            "fill_price": [p for _, p, _ in rows],
            "adv_dollar": [a for _, _, a in rows],
        }
    )


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


def test_nan_fill_price_is_zero():
    # a NaN fill price is data corruption -> 0.0 sentinel, not a silently
    # understated figure (pandas .sum() would otherwise skip the NaN).
    trades = _ledger([(100, 10.0), (100, float("nan"))])
    assert annualized_turnover(trades, _flat_equity(10_000.0, 252)) == 0.0


# --- capacity (gap 2c): model-free participation + impact-adjusted ceiling ----


def test_capacity_empty_ledger_is_none():
    rep = capacity_report(pd.DataFrame(columns=["qty", "fill_price"]), _flat_equity(1e4, 252))
    assert rep.binding == "none"
    assert rep.capacity_aum == 0.0
    assert rep.n_fills_scored == 0


def test_capacity_missing_adv_column_is_unscored():
    # a legacy ledger without adv_dollar cannot be scored -> none, not a crash.
    trades = _ledger([(100, 10.0)])
    rep = capacity_report(trades, _flat_equity(1e4, 252))
    assert rep.n_fills_scored == 0 and rep.binding == "none"


def test_capacity_participation_hand_calc():
    # one fill: notional=1000, adv=10000 -> participation 0.10. mean_equity 10000,
    # cap 0.10 -> participation_capacity = 10000 * (0.10 / 0.10) = 10000 (already at cap).
    trades = _ledger_adv([(100, 10.0, 10_000.0)])
    rep = capacity_report(
        trades, _flat_equity(1e4, 252), max_participation=0.10, impact_coef_bps=0.0
    )
    assert math.isclose(rep.p95_participation, 0.10, rel_tol=1e-9)
    assert math.isclose(rep.participation_capacity, 10_000.0, rel_tol=1e-9)
    # impact_coef 0 -> impact never binds -> participation is the binding ceiling.
    assert rep.binding == "participation"
    assert math.isclose(rep.capacity_aum, 10_000.0, rel_tol=1e-9)


def test_capacity_room_to_grow_when_below_cap():
    # participation 0.01 vs cap 0.10 -> can grow ~10x before hitting the cap.
    trades = _ledger_adv([(100, 1.0, 10_000.0)])  # notional 100 / adv 10000 = 0.01
    rep = capacity_report(
        trades, _flat_equity(1e4, 252), max_participation=0.10, impact_coef_bps=0.0
    )
    assert math.isclose(rep.participation_capacity, 1e5, rel_tol=1e-9)


def test_capacity_impact_hand_calc_and_binding():
    # one fill notional=1000, adv=10000, coef=100bps, budget=100bps, 252d (ann=1):
    #   impact$ = 1000 * 100*sqrt(0.1)/1e4 = 3.16228; g0 = 3.16228/10000 = 3.16228e-4
    #   k = (0.01 / 3.16228e-4)^2 = 1000 -> impact_capacity = 1e7
    # participation_capacity = 10000 -> participation binds.
    trades = _ledger_adv([(100, 10.0, 10_000.0)])
    rep = capacity_report(
        trades,
        _flat_equity(1e4, 252),
        max_participation=0.10,
        impact_coef_bps=100.0,
        impact_budget_bps=100.0,
    )
    assert math.isclose(rep.impact_capacity, 1e7, rel_tol=1e-6)
    assert rep.binding == "participation"
    assert math.isclose(rep.capacity_aum, 1e4, rel_tol=1e-9)


def test_capacity_impact_can_bind_before_participation():
    # tiny participation (room to grow) but a punishing impact coefficient makes
    # impact the binding constraint.
    trades = _ledger_adv([(100, 1.0, 10_000.0)])  # participation 0.01
    rep = capacity_report(
        trades,
        _flat_equity(1e4, 252),
        max_participation=0.10,
        impact_coef_bps=50_000.0,
        impact_budget_bps=50.0,
    )
    assert rep.binding == "impact"
    assert rep.capacity_aum == rep.impact_capacity
    assert rep.impact_capacity < rep.participation_capacity


def test_capacity_p95_is_robust_to_a_single_illiquid_outlier():
    # 99 liquid fills (participation 0.01) + 1 illiquid (participation 1.0). The
    # p95 ceiling should track the liquid mass, not the lone outlier.
    rows = [(100, 1.0, 1e6)] * 99 + [(100, 1.0, 100.0)]  # last: notional 100 / adv 100 = 1.0
    rep = capacity_report(
        _ledger_adv(rows), _flat_equity(1e6, 252), max_participation=0.10, impact_coef_bps=0.0
    )
    assert math.isclose(rep.max_participation, 1.0, rel_tol=1e-9)
    assert rep.p95_participation < 0.05  # outlier excluded from the headline ceiling
    assert rep.n_fills_scored == 100


def test_capacity_homogeneous_in_scale():
    # scaling notional and equity by the same factor leaves participation (and so
    # the capacity multiple) unchanged; the AUM ceiling scales with equity.
    a = capacity_report(
        _ledger_adv([(100, 10.0, 1e5)]), _flat_equity(1e4, 252), impact_coef_bps=0.0
    )
    b = capacity_report(
        _ledger_adv([(100, 100.0, 1e6)]), _flat_equity(1e5, 252), impact_coef_bps=0.0
    )
    assert math.isclose(b.participation_capacity, a.participation_capacity * 10.0, rel_tol=1e-9)
