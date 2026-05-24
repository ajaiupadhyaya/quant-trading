"""Tests for the strategy base + registry."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.strategies import REGISTRY, list_strategies, register
from quant.strategies.base import Strategy, StrategySpec


@register
class _ToyStrategy(Strategy):
    spec = StrategySpec(
        slug="toy",
        name="Toy Strategy (test only)",
        description="A placeholder used in tests.",
        universe=["AAPL", "MSFT"],
        rebalance_frequency="daily",
    )

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAPL": 1.0, "MSFT": -1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAPL": 10, "MSFT": -10}


def test_registry_contains_decorated_class() -> None:
    assert "toy" in REGISTRY
    assert REGISTRY["toy"] is _ToyStrategy


def test_list_strategies_returns_specs() -> None:
    specs = list_strategies()
    slugs = [s.slug for s in specs]
    assert "toy" in slugs


def test_strategy_can_be_instantiated_and_called() -> None:
    s = _ToyStrategy()
    signals = s.generate_signals(date(2026, 5, 23))
    assert signals.loc["AAPL"] == 1.0
    targets = s.target_positions(date(2026, 5, 23), equity=100_000)
    assert targets == {"AAPL": 10, "MSFT": -10}


def test_register_rejects_duplicate_slug() -> None:
    class _Dup(Strategy):
        spec = StrategySpec(
            slug="toy",  # duplicate
            name="dup",
            description="",
            universe=[],
            rebalance_frequency="daily",
        )

        def generate_signals(self, asof: date) -> pd.Series:
            return pd.Series()

        def target_positions(self, asof: date, equity: float) -> dict[str, int]:
            return {}

    with pytest.raises(ValueError, match="already registered"):
        register(_Dup)
