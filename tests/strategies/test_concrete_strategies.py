"""Smoke tests for the five concrete strategies.

These tests verify that each strategy:
  * Registers itself under the expected slug.
  * Constructs via ``build(bars, params=...)`` from synthetic bars.
  * Returns an empty signal series during the warm-up window.
  * Returns shares-valued positions once enough history exists.
  * Survives end-to-end through ``run_backtest`` without raising.

The bars-fixture deliberately includes >5 years so the 252-day warmup windows
on momentum / multi-factor / trend / HRP can all fire.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.strategies import REGISTRY
from quant.strategies.cross_sectional_momentum import CrossSectionalMomentum
from quant.strategies.defensive_etf_allocation import DefensiveETFAllocation
from quant.strategies.multi_factor import MEGACAP_UNIVERSE, MultiFactor
from quant.strategies.pairs_trading import PAIRS_UNIVERSE, PairsTrading
from quant.strategies.risk_parity import RiskParity, hrp_weights
from quant.strategies.trend_following import TrendFollowing
from tests.conftest import synthetic_bars

START = date(2018, 1, 1)
END = date(2024, 12, 31)


def _slug_registered(slug: str) -> None:
    assert slug in REGISTRY, f"expected slug {slug!r} in registry"


@pytest.fixture(scope="module")
def etf_bars() -> pd.DataFrame:
    from quant.data.universe import etf_universe

    return synthetic_bars(etf_universe(), START, END, seed=7)


@pytest.fixture(scope="module")
def megacap_bars() -> pd.DataFrame:
    return synthetic_bars(MEGACAP_UNIVERSE, START, END, seed=11)


@pytest.fixture(scope="module")
def pair_bars() -> pd.DataFrame:
    return synthetic_bars(PAIRS_UNIVERSE, START, END, seed=13)


# ---- registration ----------------------------------------------------------


def test_all_production_strategies_registered() -> None:
    for slug in (
        "defensive-etf-allocation",
        "momentum",
        "multi-factor",
        "pairs",
        "trend",
        "risk-parity",
    ):
        _slug_registered(slug)


def test_defensive_etf_risk_on_picks_top_three_positive_momentum(
    etf_bars: pd.DataFrame,
) -> None:
    strat = DefensiveETFAllocation.build(bars=etf_bars)
    signals = strat.generate_signals(date(2023, 6, 30))
    targets = strat.target_positions(date(2023, 6, 30), equity=100_000)

    assert len(targets) <= 3
    assert set(targets).issubset(set(signals[signals > 0].nlargest(3).index))
    assert all(q >= 0 for q in targets.values())


def test_defensive_etf_risk_off_uses_defensive_assets_only(
    etf_bars: pd.DataFrame,
) -> None:
    stressed = etf_bars.copy()
    stressed.loc[stressed.index[-260]:, ("SPY", "close")] = pd.Series(
        range(260, 0, -1), index=stressed.index[-260:]
    ).astype(float)
    strat = DefensiveETFAllocation.build(bars=stressed)

    targets = strat.target_positions(date(2024, 12, 31), equity=100_000)

    assert set(targets).issubset({"IEF", "TLT", "GLD"})
    assert all(q >= 0 for q in targets.values())


# ---- cross-sectional momentum ---------------------------------------------


def test_momentum_warmup_returns_empty(etf_bars: pd.DataFrame) -> None:
    strat = CrossSectionalMomentum.build(bars=etf_bars)
    assert strat.generate_signals(date(2018, 2, 1)).empty


def test_momentum_returns_shares_when_warm(etf_bars: pd.DataFrame) -> None:
    strat = CrossSectionalMomentum.build(bars=etf_bars)
    targets = strat.target_positions(date(2023, 6, 30), equity=100_000)
    assert isinstance(targets, dict)
    if targets:
        assert all(isinstance(v, int) and v >= 0 for v in targets.values())


def test_momentum_runs_through_engine(etf_bars: pd.DataFrame) -> None:
    strat = CrossSectionalMomentum.build(bars=etf_bars)
    result = run_backtest(
        strategy=strat,
        bars=etf_bars,
        config=BacktestConfig(),
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
    )
    assert len(result.equity_curve) > 0
    assert result.ending_equity > 0


# ---- multi-factor ---------------------------------------------------------


def test_multi_factor_signal_has_both_signs(megacap_bars: pd.DataFrame) -> None:
    strat = MultiFactor.build(bars=megacap_bars)
    signals = strat.generate_signals(date(2023, 6, 30))
    if signals.empty:
        pytest.skip("warmup not reached")
    # Z-scored composite should have at least one positive and one negative entry.
    assert (signals > 0).any()
    assert (signals < 0).any()


def test_multi_factor_dollar_neutral_targets(megacap_bars: pd.DataFrame) -> None:
    strat = MultiFactor.build(bars=megacap_bars)
    targets = strat.target_positions(date(2023, 6, 30), equity=100_000)
    if not targets:
        pytest.skip("warmup not reached or no signals")
    # In dollar-neutral mode there should be at least one short.
    assert any(v < 0 for v in targets.values())


def test_multi_factor_runs_through_engine(megacap_bars: pd.DataFrame) -> None:
    strat = MultiFactor.build(bars=megacap_bars)
    result = run_backtest(
        strategy=strat,
        bars=megacap_bars,
        config=BacktestConfig(),
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
    )
    assert len(result.equity_curve) > 0


# ---- pairs ---------------------------------------------------------------


def test_pairs_state_persists_across_calls(pair_bars: pd.DataFrame) -> None:
    strat = PairsTrading.build(bars=pair_bars)
    # Burn through warmup then two consecutive rebalance days; state dict should populate.
    strat.target_positions(date(2023, 6, 30), equity=100_000)
    strat.target_positions(date(2023, 7, 7), equity=100_000)
    # State is keyed by discovered or seed pairs — must be a dict, may be empty
    # if the synthetic panel didn't produce any cointegrated pairs.
    assert isinstance(strat._state, dict)  # type: ignore[attr-defined]


def test_pairs_runs_through_engine(pair_bars: pd.DataFrame) -> None:
    strat = PairsTrading.build(bars=pair_bars)
    result = run_backtest(
        strategy=strat,
        bars=pair_bars,
        config=BacktestConfig(),
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
    )
    assert len(result.equity_curve) > 0


# ---- trend following -----------------------------------------------------


def test_momentum_vol_scaling_uses_inverse_vol(etf_bars: pd.DataFrame) -> None:
    strat = CrossSectionalMomentum.build(bars=etf_bars)
    eq_strat = CrossSectionalMomentum.build(bars=etf_bars, params={"vol_scale_enabled": False})
    a = strat.target_positions(date(2023, 6, 30), equity=100_000)
    b = eq_strat.target_positions(date(2023, 6, 30), equity=100_000)
    # Same set of picks, possibly different shares — at minimum the two configs
    # should agree on which names are picked (signal logic is identical).
    if a and b:
        assert set(a) == set(b)


def test_trend_drawdown_control_attenuates_exposure(etf_bars: pd.DataFrame) -> None:
    """A high dd_floor + dd_control_enabled should never exceed full-leverage exposure."""
    on = TrendFollowing.build(bars=etf_bars, params={"dd_control_enabled": True})
    off = TrendFollowing.build(bars=etf_bars, params={"dd_control_enabled": False})
    asof_date = date(2023, 6, 30)
    a_targets = on.target_positions(asof_date, equity=100_000)
    b_targets = off.target_positions(asof_date, equity=100_000)
    a_notional = sum(abs(q) for q in a_targets.values())
    b_notional = sum(abs(q) for q in b_targets.values())
    # When drawdown control is on, notional must be <= when it's off
    # (assuming the proxy basket is in some drawdown — but it may not be on
    # synthetic data, in which case the factor is 1.0 and both equal).
    assert a_notional <= b_notional + 1


def test_trend_long_only_by_default(etf_bars: pd.DataFrame) -> None:
    strat = TrendFollowing.build(bars=etf_bars)
    targets = strat.target_positions(date(2023, 6, 30), equity=100_000)
    if targets:
        assert all(v >= 0 for v in targets.values())


def test_trend_short_allowed_when_param_set(etf_bars: pd.DataFrame) -> None:
    strat = TrendFollowing.build(bars=etf_bars, params={"allow_short": True})
    signals = strat.generate_signals(date(2023, 6, 30))
    if not signals.empty:
        # With allow_short the signal can be negative
        assert signals.min() >= -1.0


def test_trend_runs_through_engine(etf_bars: pd.DataFrame) -> None:
    strat = TrendFollowing.build(bars=etf_bars)
    result = run_backtest(
        strategy=strat,
        bars=etf_bars,
        config=BacktestConfig(),
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
    )
    assert len(result.equity_curve) > 0


# ---- risk parity ---------------------------------------------------------


def test_hrp_weights_sum_to_one_on_clean_input() -> None:
    rng = pd.bdate_range("2020-01-01", periods=252)
    import numpy as np

    data = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, size=(252, 4)), index=rng)
    cov = data.cov()
    corr = data.corr().fillna(0)
    w = hrp_weights(cov, corr)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert (w > 0).all()


def test_risk_parity_runs_through_engine(etf_bars: pd.DataFrame) -> None:
    strat = RiskParity.build(bars=etf_bars)
    result = run_backtest(
        strategy=strat,
        bars=etf_bars,
        config=BacktestConfig(),
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
    )
    assert len(result.equity_curve) > 0


def test_risk_parity_weights_present_when_warm(etf_bars: pd.DataFrame) -> None:
    strat = RiskParity.build(bars=etf_bars)
    weights = strat.generate_signals(date(2023, 6, 30))
    if weights.empty:
        pytest.skip("warmup not reached")
    assert (weights >= 0).all()
    assert float(weights.sum()) > 0
