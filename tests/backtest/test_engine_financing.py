"""Integration tests: borrow/financing charged through run_backtest."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.strategies.base import Strategy, StrategySpec
from tests.conftest import EqualWeightStrategy


def _flat_bars(symbols: list[str], price: float = 100.0) -> pd.DataFrame:
    """Bars where every field equals `price`, every business day in 2024-Q1."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = pd.DataFrame(
            {f: np.full(len(dates), price) for f in ("open", "high", "low", "close")}
            | {"volume": np.full(len(dates), 1_000_000, dtype=np.int64)},
            index=dates,
        )
        df.index.name = "timestamp"
        frames[sym] = df
    return pd.concat(frames, axis=1)


class _FixedShortStrategy(Strategy):
    """Test-only: hold a fixed -100 share short in AAA at every rebalance."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="fixed-short-test",
        name="Fixed Short (test)",
        description="Test fixture: constant 100-share short in AAA.",
        universe=["AAA"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def __init__(self, bars: pd.DataFrame) -> None:
        super().__init__(params=None)
        self._bars = bars

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": -1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": -100}


def _short_cfg(borrow: float, fin: float) -> BacktestConfig:
    return BacktestConfig(
        starting_equity=1_000_000.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        annual_borrow_bps=borrow,
        annual_financing_bps=fin,
        execution="close",
    )


def test_borrow_matches_expected_and_drains_equity():
    bars = _flat_bars(["AAA"])
    strat = _FixedShortStrategy(bars)
    res = run_backtest(strat, bars, _short_cfg(50.0, 0.0), date(2024, 1, 2), date(2024, 3, 29))
    idx = res.equity_curve.index
    days = (idx[-1] - idx[0]).days  # consecutive-bar gaps telescope to this span
    expected = 100 * 100.0 * (50.0 / 1e4) * (days / 365.0)
    assert res.metadata["financing_cost_total"] == pytest.approx(expected, rel=1e-9)
    assert res.metadata["margin_financing_cost"] == 0.0  # short proceeds keep cash positive

    res0 = run_backtest(strat, bars, _short_cfg(0.0, 0.0), date(2024, 1, 2), date(2024, 3, 29))
    assert res0.metadata["financing_cost_total"] == 0.0
    assert res.ending_equity == pytest.approx(res0.ending_equity - expected, rel=1e-9)


def test_financing_uses_prior_close_not_today_pit():
    # Spike the LAST bar's close to $200. A PIT charge uses the PRIOR close ($100)
    # for every accrual, so the total stays $100-based; using today's close would
    # inflate the final day's charge.
    bars = _flat_bars(["AAA"])
    last = bars.index[-1]
    bars.loc[last, ("AAA", "close")] = 200.0
    strat = _FixedShortStrategy(bars)
    res = run_backtest(strat, bars, _short_cfg(50.0, 0.0), date(2024, 1, 2), date(2024, 3, 29))
    idx = res.equity_curve.index
    days = (idx[-1] - idx[0]).days
    expected = 100 * 100.0 * (50.0 / 1e4) * (days / 365.0)
    assert res.metadata["financing_cost_total"] == pytest.approx(expected, rel=1e-9)


def test_long_only_default_rates_zero_financing(make_bars):
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    res = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    assert res.metadata["financing_cost_total"] == 0.0
