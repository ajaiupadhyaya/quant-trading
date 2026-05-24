"""Strategy base class parameter merging."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd

from quant.strategies.base import Strategy, StrategySpec


class _ParamsToyStrategy(Strategy):
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="toy-params",
        name="Toy Params",
        description="Toy strategy used by parameter-merging tests.",
        universe=["AAPL"],
        rebalance_frequency="daily",
    )
    default_params: ClassVar[dict[str, object]] = {"lookback": 10, "scale": 1.0}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAPL": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAPL": 1}


class _NestedDefaultsStrategy(Strategy):
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="toy-nested",
        name="Toy Nested",
        description="Strategy with a nested-dict default param.",
        universe=["AAPL"],
        rebalance_frequency="daily",
    )
    default_params: ClassVar[dict[str, object]] = {"thresholds": {"entry": 1.0, "exit": 0.5}}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAPL": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {}


def test_default_params_used_when_none_passed() -> None:
    s = _ParamsToyStrategy()
    assert s.params == {"lookback": 10, "scale": 1.0}


def test_params_override_defaults() -> None:
    s = _ParamsToyStrategy(params={"lookback": 20})
    assert s.params == {"lookback": 20, "scale": 1.0}


def test_extra_params_pass_through() -> None:
    s = _ParamsToyStrategy(params={"new_knob": True})
    assert s.params["new_knob"] is True
    assert s.params["lookback"] == 10


def test_default_params_not_mutated_by_instance() -> None:
    s = _ParamsToyStrategy(params={"lookback": 99})
    assert s.params["lookback"] == 99
    assert _ParamsToyStrategy.default_params["lookback"] == 10


def test_empty_params_dict_behaves_like_none() -> None:
    s = _ParamsToyStrategy(params={})
    assert s.params == {"lookback": 10, "scale": 1.0}


def test_nested_default_params_deep_copied() -> None:
    """Mutating a nested value on the instance must not leak into the class default."""
    s = _NestedDefaultsStrategy()
    # type: ignore[index]  -- the test verifies dict mutation semantics
    s.params["thresholds"]["entry"] = 999.0  # type: ignore[index]
    assert _NestedDefaultsStrategy.default_params["thresholds"]["entry"] == 1.0  # type: ignore[index]
    # And a fresh instance reads the original default, not the leaked value.
    s2 = _NestedDefaultsStrategy()
    assert s2.params["thresholds"]["entry"] == 1.0  # type: ignore[index]


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
