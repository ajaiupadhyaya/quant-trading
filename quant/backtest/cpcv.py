"""Combinatorial Purged Cross-Validation (Lopez de Prado 2018).

Splits the timeline into N contiguous groups and runs the strategy on every
C(N, K) combination of K test groups, with an embargo around each test
boundary to mitigate serial-correlation leakage. The aggregated per-path
Sharpe distribution informs the Deflated Sharpe Ratio's trial-count term.

This implementation evaluates a strategy with *fixed* params on each test
segment — params should have been pre-selected by walk-forward. CPCV here
measures robustness of OOS Sharpe to test-set placement, not a search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, run_backtest
from quant.backtest.metrics import sharpe

if TYPE_CHECKING:
    from quant.strategies.base import Strategy


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    k_test: int = 2
    embargo_days: int = 5


@dataclass(frozen=True)
class CPCVResult:
    path_sharpes: np.ndarray  # one Sharpe per combination, shape (C(N,K),)
    n_groups: int
    k_test: int


StrategyFactory = Callable[[dict[str, Any], pd.DataFrame], "Strategy"]


def make_groups(index: pd.DatetimeIndex, n_groups: int) -> list[list[pd.Timestamp]]:
    """Partition ``index`` into ``n_groups`` contiguous, non-overlapping groups."""
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}")
    # Require at least 2 bars per group so each test segment is meaningful.
    if len(index) < n_groups * 2:
        return []
    # np.array_split divides as evenly as possible; remainder distributed to leading groups.
    splits = np.array_split(np.asarray(index), n_groups)
    return [list(pd.DatetimeIndex(s)) for s in splits]


def iter_combinations(n_groups: int, k_test: int):
    """Yield each combination of k_test group indices out of range(n_groups)."""
    if k_test <= 0 or k_test >= n_groups:
        raise ValueError(f"k_test must satisfy 0 < k_test < n_groups; got {k_test}, {n_groups}")
    yield from combinations(range(n_groups), k_test)


def _test_window_from_groups(
    groups: list[list[pd.Timestamp]], test_indices: tuple[int, ...]
) -> tuple[date, date]:
    """Concatenate the chosen groups into a single [start, end] date window.

    Returns the contiguous span between the earliest and latest timestamp in the
    chosen groups; for non-contiguous group selections, the inner gap is left
    inside the window (the engine restricts via bars anyway).
    """
    timestamps: list[pd.Timestamp] = []
    for i in test_indices:
        timestamps.extend(groups[i])
    if not timestamps:
        raise ValueError("test_indices selected empty groups")
    return min(timestamps).date(), max(timestamps).date()


def run_cpcv(
    strategy_factory: StrategyFactory,
    params: dict[str, Any],
    bars: pd.DataFrame,
    start: date,
    end: date,
    backtest_config: BacktestConfig,
    cpcv_config: CPCVConfig,
) -> CPCVResult:
    """Run the strategy on each combinatorial test split; return path Sharpes."""
    mask = (bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end))
    window_index = pd.DatetimeIndex(bars.index[mask])

    groups = make_groups(window_index, cpcv_config.n_groups)
    if not groups:
        return CPCVResult(
            path_sharpes=np.array([], dtype=float),
            n_groups=cpcv_config.n_groups,
            k_test=cpcv_config.k_test,
        )

    embargo = timedelta(days=cpcv_config.embargo_days)
    path_sharpes: list[float] = []

    for combo in iter_combinations(cpcv_config.n_groups, cpcv_config.k_test):
        test_start, test_end = _test_window_from_groups(groups, combo)
        # Embargo shrinks the test window away from training boundaries.
        test_start = test_start + embargo
        test_end = test_end - embargo
        if test_end <= test_start:
            continue

        strat = strategy_factory(params, bars)
        bt = run_backtest(strat, bars, backtest_config, test_start, test_end)
        path_sharpes.append(sharpe(bt.returns))

    return CPCVResult(
        path_sharpes=np.asarray(path_sharpes, dtype=float),
        n_groups=cpcv_config.n_groups,
        k_test=cpcv_config.k_test,
    )
