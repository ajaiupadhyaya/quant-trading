"""Tests for run_walkforward."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestResult
from quant.backtest.walkforward import WalkforwardResult, run_walkforward
from quant.strategies.base import Strategy
from tests.conftest import EqualWeightStrategy


def _factory(params: dict[str, Any], bars: pd.DataFrame) -> Strategy:
    return EqualWeightStrategy(bars=bars, params=params)


# Shrinking the test window from 11y → 7y reduces window count from 11 → 3 while
# still exercising the multi-window stitch path. Cut test runtime by ~3.5x.
_WF_START = date(2010, 1, 1)
_WF_END = date(2017, 1, 1)


def test_oos_curve_starts_at_first_test_window(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], _WF_START, _WF_END, seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},  # single-point grid
        bars=bars,
        start=_WF_START,
        end=_WF_END,
        config=BacktestConfig(slippage_bps=0.0),
    )
    assert isinstance(result, WalkforwardResult)
    assert result.oos_equity_curve.index.min() >= pd.Timestamp("2015-01-01")


def test_per_window_params_present(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], _WF_START, _WF_END, seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=_WF_START,
        end=_WF_END,
        config=BacktestConfig(),
    )
    assert len(result.per_window_params) > 0
    for _window, params in result.per_window_params:
        assert params == {"_dummy": 1}


def test_combined_result_has_full_oos_history(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], _WF_START, _WF_END, seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=_WF_START,
        end=_WF_END,
        config=BacktestConfig(),
    )
    assert isinstance(result.combined_result, BacktestResult)
    # The combined result's equity_curve should equal the stitched OOS curve.
    pd.testing.assert_series_equal(
        result.combined_result.equity_curve, result.oos_equity_curve, check_names=False
    )


def test_oos_curve_monotone_chronological(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], _WF_START, _WF_END, seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=_WF_START,
        end=_WF_END,
        config=BacktestConfig(),
    )
    idx = result.oos_equity_curve.index
    # Stitching must produce strictly increasing dates (no duplicates).
    assert (idx[1:] > idx[:-1]).all()
    # And must exercise multi-window stitching (at least 2 windows present).
    assert len(result.per_window_params) >= 2


def test_no_windows_returns_empty_result(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2020, 1, 1), date(2021, 12, 31), seed=0)
    # 2y of data, default 5y train + 1y test → no fit-able window.
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2020, 1, 1),
        end=date(2021, 12, 31),
        config=BacktestConfig(),
    )
    assert len(result.oos_equity_curve) == 0
    assert len(result.per_window_params) == 0


def test_grid_trial_sharpes_captures_every_window_combo(
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    """Walk-forward records one trial Sharpe per (contributing window, grid combo).

    These are the true model-selection trials the Deflated Sharpe must deflate
    against (not the CPCV resample paths).
    """
    import numpy as np

    bars = make_bars(["AAA", "BBB"], _WF_START, _WF_END, seed=0)
    grid = {"_dummy": [1, 2, 3]}  # 3 combos
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid=grid,
        bars=bars,
        start=_WF_START,
        end=_WF_END,
        config=BacktestConfig(),
    )
    n_windows = len(result.per_window_params)
    assert n_windows > 0
    assert isinstance(result.grid_trial_sharpes, np.ndarray)
    assert len(result.grid_trial_sharpes) == n_windows * 3


def test_grid_trial_sharpes_empty_when_no_windows(
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    import numpy as np

    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2011, 1, 1), seed=0)
    result = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2011, 1, 1),  # < train_years -> no windows
        config=BacktestConfig(),
    )
    assert len(result.per_window_params) == 0
    assert isinstance(result.grid_trial_sharpes, np.ndarray)
    assert len(result.grid_trial_sharpes) == 0
