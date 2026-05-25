"""Tests for the per-strategy walk-forward parameter grids.

Verifies that:
- Every registered strategy declares a non-empty ``param_grid``.
- Every key in ``param_grid`` is also a key in ``default_params`` (so grid-search
  doesn't silently feed unknown knobs).
- A small walk-forward over synthetic bars actually exercises the grid and
  picks a winner per training window without raising.
"""

from __future__ import annotations

from datetime import date

import pytest

from quant.backtest.engine import BacktestConfig
from quant.backtest.walkforward import iter_windows, run_walkforward, select_best_params
from quant.strategies import REGISTRY
from tests.conftest import synthetic_bars


@pytest.mark.parametrize("slug", sorted(REGISTRY.keys()))
def test_strategy_declares_non_empty_grid(slug: str) -> None:
    cls = REGISTRY[slug]
    # Test-only fixtures intentionally skip the grid; concrete strategies do not.
    if slug in {"toy", "toy-params", "toy-nested", "no-defaults", "equal-weight-test"}:
        pytest.skip("test fixture strategy")
    assert cls.param_grid, f"{slug} must declare a non-empty param_grid"


@pytest.mark.parametrize("slug", sorted(REGISTRY.keys()))
def test_param_grid_keys_subset_of_defaults(slug: str) -> None:
    cls = REGISTRY[slug]
    if not cls.param_grid:
        pytest.skip("no grid declared")
    unknown = set(cls.param_grid) - set(cls.default_params)
    assert not unknown, f"{slug} grid keys not in default_params: {sorted(unknown)}"


def test_walkforward_uses_grid_and_returns_chosen_params() -> None:
    """A 3-year walk-forward with a 3-knob grid picks per-window params."""
    cls = REGISTRY["momentum"]
    bars = synthetic_bars(list(cls.spec.universe), date(2018, 1, 1), date(2023, 12, 31), seed=3)

    def factory(params, bars_for_strategy):  # type: ignore[no-untyped-def]
        return cls.build(bars=bars_for_strategy, params=params)

    # Use a tiny 2-value grid to keep this test fast.
    grid = {"top_pct": [0.25, 0.40]}
    result = run_walkforward(
        strategy_factory=factory,
        param_grid=grid,
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2023, 12, 31),
        config=BacktestConfig(),
        train_years=3,
        test_years=1,
        step_months=12,
    )
    # At least one window should have been produced + given chosen params.
    assert result.per_window_params, "walk-forward should produce at least one window"
    for _window, chosen in result.per_window_params:
        assert "top_pct" in chosen
        assert chosen["top_pct"] in grid["top_pct"]


def test_select_best_params_returns_member_of_grid() -> None:
    cls = REGISTRY["risk-parity"]
    bars = synthetic_bars(list(cls.spec.universe), date(2018, 1, 1), date(2023, 12, 31), seed=5)

    def factory(params, bars_for_strategy):  # type: ignore[no-untyped-def]
        return cls.build(bars=bars_for_strategy, params=params)

    grid = {"vol_target_annual": [0.08, 0.12]}
    win = next(iter_windows(date(2018, 1, 1), date(2023, 12, 31), train_years=3, test_years=1))
    best = select_best_params(
        strategy_factory=factory,
        param_grid=grid,
        bars=bars,
        window=win,
        config=BacktestConfig(),
    )
    assert "vol_target_annual" in best
    assert best["vol_target_annual"] in grid["vol_target_annual"]
