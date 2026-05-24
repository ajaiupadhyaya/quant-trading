"""Strategy base class parameter merging."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd

from quant.strategies.base import Strategy, StrategySpec


class _ToyStrategy(Strategy):
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="toy",
        name="Toy",
        description="Toy strategy used in tests.",
        universe=["AAPL"],
        rebalance_frequency="daily",
    )
    default_params: ClassVar[dict[str, object]] = {"lookback": 10, "scale": 1.0}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAPL": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAPL": 1}


def test_default_params_used_when_none_passed() -> None:
    s = _ToyStrategy()
    assert s.params == {"lookback": 10, "scale": 1.0}


def test_params_override_defaults() -> None:
    s = _ToyStrategy(params={"lookback": 20})
    assert s.params == {"lookback": 20, "scale": 1.0}


def test_extra_params_pass_through() -> None:
    s = _ToyStrategy(params={"new_knob": True})
    assert s.params["new_knob"] is True
    assert s.params["lookback"] == 10


def test_default_params_not_mutated_by_instance() -> None:
    s = _ToyStrategy(params={"lookback": 99})
    assert s.params["lookback"] == 99
    assert _ToyStrategy.default_params["lookback"] == 10


def test_strategy_without_default_params_works() -> None:
    """A Strategy subclass that doesn't declare default_params still instantiates."""

    class _NoDefaults(Strategy):
        spec: ClassVar[StrategySpec] = StrategySpec(
            slug="no-defaults",
            name="No Defaults",
            description="-",
            universe=["AAPL"],
            rebalance_frequency="daily",
        )

        def generate_signals(self, asof: date) -> pd.Series:
            return pd.Series(dtype=float)

        def target_positions(self, asof: date, equity: float) -> dict[str, int]:
            return {}

    s = _NoDefaults()
    assert s.params == {}
