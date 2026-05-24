"""Tests for select_best_params."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, ClassVar

import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig
from quant.backtest.walkforward import WalkforwardWindow, select_best_params
from quant.strategies.base import Strategy, StrategySpec
from tests.conftest import EqualWeightStrategy


class _TiltedStrategy(Strategy):
    """Long-only allocates 100% to "AAA" when params['tilt'] = 'aaa' else "BBB"."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="tilted-test",
        name="Tilted (test)",
        description="-",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, Any]] = {"tilt": "aaa"}

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._bars = bars

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0, "BBB": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        ts = pd.Timestamp(asof)
        if ts not in self._bars.index:
            return {}
        sym = "AAA" if self.params["tilt"] == "aaa" else "BBB"
        price = float(self._bars[(sym, "close")].loc[ts])
        return {sym: int(equity // price)}


def test_select_best_params_picks_higher_sharpe(make_bars: Callable[..., pd.DataFrame]) -> None:
    # Construct bars where AAA strongly trends up and BBB trends down.
    bars_aaa = make_bars(["AAA"], date(2020, 1, 1), date(2024, 12, 31), seed=0, drift=0.002)
    bars_bbb = make_bars(["BBB"], date(2020, 1, 1), date(2024, 12, 31), seed=0, drift=-0.001)
    bars = pd.concat([bars_aaa, bars_bbb], axis=1)

    def factory(params: dict[str, Any], bars_for_strategy: pd.DataFrame) -> Strategy:
        return _TiltedStrategy(bars=bars_for_strategy, params=params)

    window = WalkforwardWindow(
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
    )

    best = select_best_params(
        strategy_factory=factory,
        param_grid={"tilt": ["aaa", "bbb"]},
        bars=bars,
        window=window,
        config=BacktestConfig(slippage_bps=0.0),
    )
    assert best == {"tilt": "aaa"}


def test_select_best_params_handles_empty_grid(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2020, 1, 1), date(2024, 12, 31), seed=0)

    def factory(params: dict[str, Any], bars_for_strategy: pd.DataFrame) -> Strategy:
        return EqualWeightStrategy(bars=bars_for_strategy, params=params)

    window = WalkforwardWindow(
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
    )
    best = select_best_params(
        strategy_factory=factory,
        param_grid={},
        bars=bars,
        window=window,
        config=BacktestConfig(),
    )
    assert best == {}


def test_select_best_params_explores_full_grid(
    make_bars: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Cartesian product: 2x3 grid → 6 engine calls."""
    bars = make_bars(["AAA", "BBB"], date(2020, 1, 1), date(2024, 12, 31), seed=0)

    call_count = 0

    def factory(params: dict[str, Any], bars_for_strategy: pd.DataFrame) -> Strategy:
        nonlocal call_count
        call_count += 1
        return EqualWeightStrategy(bars=bars_for_strategy, params=params)

    window = WalkforwardWindow(
        train_start=date(2020, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
    )
    select_best_params(
        strategy_factory=factory,
        param_grid={"a": [1, 2], "b": [10, 20, 30]},
        bars=bars,
        window=window,
        config=BacktestConfig(),
    )
    assert call_count == 6
