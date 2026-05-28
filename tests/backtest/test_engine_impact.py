"""Integration tests: market impact charged through run_backtest."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.strategies.base import Strategy, StrategySpec


def _bars(symbol: str, price: float, volume: int) -> pd.DataFrame:
    """Flat-price bars for 2024-Q1 with a constant per-bar volume."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    df = pd.DataFrame(
        {f: np.full(len(dates), price) for f in ("open", "high", "low", "close")}
        | {"volume": np.full(len(dates), volume, dtype=np.int64)},
        index=dates,
    )
    df.index.name = "timestamp"
    return pd.concat({symbol: df}, axis=1)


class _FixedLongStrategy(Strategy):
    """Test-only: hold a fixed 1,000-share long in AAA at every rebalance."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="fixed-long-test",
        name="Fixed Long (test)",
        description="Test fixture: constant 1,000-share long in AAA.",
        universe=["AAA"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def __init__(self, bars: pd.DataFrame) -> None:
        super().__init__(params=None)
        self._bars = bars

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 1000}


def _cfg(impact_coef_bps: float, adv_window: int = 21) -> BacktestConfig:
    # next_open execution so the first rebalance (bar 1) fills on bar 2, which has
    # bar 1 as strictly-prior history for ADV (a close-execution fill on bar 1 would
    # have no prior history -> ADV 0 -> impact 0). Zero spread/commission/financing
    # so impact is the only cost in play.
    return BacktestConfig(
        starting_equity=10_000_000.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        annual_borrow_bps=0.0,
        annual_financing_bps=0.0,
        impact_coef_bps=impact_coef_bps,
        adv_window=adv_window,
        execution="next_open",
    )


def test_low_adv_costs_more_than_high_adv():
    low = run_backtest(
        _FixedLongStrategy(_bars("AAA", 100.0, 5_000)),
        _bars("AAA", 100.0, 5_000),
        _cfg(100.0),
        date(2024, 1, 2),
        date(2024, 3, 29),
    )
    high = run_backtest(
        _FixedLongStrategy(_bars("AAA", 100.0, 50_000_000)),
        _bars("AAA", 100.0, 50_000_000),
        _cfg(100.0),
        date(2024, 1, 2),
        date(2024, 3, 29),
    )
    assert low.ending_equity < high.ending_equity


def test_zero_coef_disables_impact():
    bars = _bars("AAA", 100.0, 5_000)
    res = run_backtest(
        _FixedLongStrategy(bars), bars, _cfg(0.0), date(2024, 1, 2), date(2024, 3, 29)
    )
    assert all(abs(eq - 10_000_000.0) < 100.0 for eq in res.equity_curve)


def test_impact_uses_prior_volume_not_fill_bar_pit():
    bars = _bars("AAA", 100.0, 5_000)
    res = run_backtest(
        _FixedLongStrategy(bars), bars, _cfg(100.0), date(2024, 1, 2), date(2024, 3, 29)
    )
    first_trade_ts = res.trades["date"].min()

    spiked = _bars("AAA", 100.0, 5_000)
    spiked.loc[first_trade_ts, ("AAA", "volume")] = 9_999_999_999
    res_spiked = run_backtest(
        _FixedLongStrategy(spiked), spiked, _cfg(100.0), date(2024, 1, 2), date(2024, 3, 29)
    )
    assert res_spiked.ending_equity == res.ending_equity
