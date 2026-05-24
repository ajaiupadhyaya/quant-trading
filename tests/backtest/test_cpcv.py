from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.backtest.cpcv import (
    CPCVConfig,
    CPCVResult,
    iter_combinations,
    make_groups,
    run_cpcv,
)
from quant.backtest.engine import BacktestConfig
from tests.conftest import synthetic_bars

from quant.strategies.base import Strategy, StrategySpec


class _EqualWeightOneShot(Strategy):
    spec = StrategySpec(
        slug="cpcv-test-eqw",
        name="EqualWeightOneShot",
        description="Test fixture",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0, "BBB": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 10, "BBB": 10}


def test_make_groups_returns_n_contiguous_index_ranges() -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-01", periods=100))
    groups = make_groups(idx, n_groups=5)
    assert len(groups) == 5
    # Contiguous, non-overlapping, covers all
    flat = [t for g in groups for t in g]
    assert flat == list(idx)
    assert all(len(g) >= 20 - 1 for g in groups)


def test_iter_combinations_yields_n_choose_k() -> None:
    from math import comb

    combos = list(iter_combinations(n_groups=6, k_test=2))
    assert len(combos) == comb(6, 2)
    # Each combination is k_test distinct group indices
    for c in combos:
        assert len(set(c)) == 2
        assert all(0 <= i < 6 for i in c)


def test_iter_combinations_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        list(iter_combinations(n_groups=4, k_test=0))
    with pytest.raises(ValueError):
        list(iter_combinations(n_groups=4, k_test=5))


def test_run_cpcv_returns_one_path_sharpe_per_combination() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2022, 12, 31))
    cfg = CPCVConfig(n_groups=4, k_test=2, embargo_days=0)
    bt_cfg = BacktestConfig(starting_equity=100_000.0)

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightOneShot(params=params)

    result = run_cpcv(
        strategy_factory=factory,
        params={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2022, 12, 31),
        backtest_config=bt_cfg,
        cpcv_config=cfg,
    )
    from math import comb

    assert isinstance(result, CPCVResult)
    assert len(result.path_sharpes) == comb(4, 2)
    assert result.n_groups == 4
    assert result.k_test == 2


def test_run_cpcv_path_sharpes_finite() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2022, 12, 31))
    cfg = CPCVConfig(n_groups=4, k_test=2, embargo_days=0)
    bt_cfg = BacktestConfig(starting_equity=100_000.0)

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightOneShot(params=params)

    result = run_cpcv(
        strategy_factory=factory,
        params={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2022, 12, 31),
        backtest_config=bt_cfg,
        cpcv_config=cfg,
    )
    assert np.all(np.isfinite(result.path_sharpes))


def test_run_cpcv_with_empty_window_returns_empty_paths() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2018, 1, 5))
    cfg = CPCVConfig(n_groups=4, k_test=2, embargo_days=0)
    bt_cfg = BacktestConfig(starting_equity=100_000.0)

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightOneShot(params=params)

    result = run_cpcv(
        strategy_factory=factory,
        params={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2018, 1, 5),
        backtest_config=bt_cfg,
        cpcv_config=cfg,
    )
    # Fewer bars than groups → empty paths
    assert len(result.path_sharpes) == 0
