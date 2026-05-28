"""Walk-forward harness: rolling train/test windows, grid search per window."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from typing import TYPE_CHECKING, Any

import pandas as pd
from dateutil.relativedelta import relativedelta

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.backtest.metrics import sharpe
from quant.util.logging import logger

if TYPE_CHECKING:
    from quant.strategies.base import Strategy


@dataclass(frozen=True)
class WalkforwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def iter_windows(
    start: date,
    end: date,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 6,
) -> Iterator[WalkforwardWindow]:
    """Yield rolling train/test windows over [start, end].

    The first window's train_start = ``start``. Each subsequent window steps the
    train_start forward by ``step_months``. A window is yielded only if its
    test_end <= ``end``.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")
    if train_years <= 0 or test_years <= 0 or step_months <= 0:
        raise ValueError("train_years, test_years, step_months must all be positive")

    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + relativedelta(years=test_years) - timedelta(days=1)
        if test_end > end:
            return
        yield WalkforwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        train_start = train_start + relativedelta(months=step_months)


StrategyFactory = Callable[[dict[str, Any], pd.DataFrame], "Strategy"]


@dataclass(frozen=True)
class WalkforwardResult:
    """Output of run_walkforward."""

    oos_equity_curve: pd.Series
    oos_returns: pd.Series
    oos_trades: pd.DataFrame
    per_window_params: list[tuple[WalkforwardWindow, dict[str, Any]]]
    combined_result: BacktestResult


def _iter_grid(param_grid: dict[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Cartesian product of ``param_grid``. Yields one dict per combo.

    An empty grid yields one empty dict (the strategy's defaults are used).
    """
    if not param_grid:
        yield {}
        return
    keys = list(param_grid.keys())
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def select_best_params(
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Sequence[Any]],
    bars: pd.DataFrame,
    window: WalkforwardWindow,
    config: BacktestConfig,
) -> dict[str, Any]:
    """Grid-search ``param_grid`` on the train window, return best by Sharpe."""
    best_params: dict[str, Any] = {}
    best_score: float = float("-inf")

    for params in _iter_grid(param_grid):
        strat = strategy_factory(params, bars)
        result = run_backtest(strat, bars, config, window.train_start, window.train_end)
        score = sharpe(result.returns)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def run_walkforward(
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Sequence[Any]],
    bars: pd.DataFrame,
    start: date,
    end: date,
    config: BacktestConfig,
    train_years: int = 5,
    test_years: int = 1,
    step_months: int = 6,
) -> WalkforwardResult:
    """Walk-forward orchestrator. Stitches per-window OOS backtests into one curve."""
    oos_equity_pieces: list[pd.Series] = []
    oos_trades_pieces: list[pd.DataFrame] = []
    per_window_params: list[tuple[WalkforwardWindow, dict[str, Any]]] = []

    cumulative_equity: float = config.starting_equity

    for window in iter_windows(start, end, train_years, test_years, step_months):
        logger.info(
            "Walk-forward: train {}..{} -> test {}..{}",
            window.train_start,
            window.train_end,
            window.test_start,
            window.test_end,
        )
        best = select_best_params(strategy_factory, param_grid, bars, window, config)

        test_config = BacktestConfig(
            starting_equity=cumulative_equity,
            slippage_bps=config.slippage_bps,
            commission_bps=config.commission_bps,
            annual_borrow_bps=config.annual_borrow_bps,
            annual_financing_bps=config.annual_financing_bps,
            execution=config.execution,
        )
        test_strat = strategy_factory(best, bars)
        test_result = run_backtest(
            test_strat, bars, test_config, window.test_start, window.test_end
        )
        if len(test_result.equity_curve) == 0:
            continue
        oos_equity_pieces.append(test_result.equity_curve)
        oos_trades_pieces.append(test_result.trades)
        per_window_params.append((window, best))
        cumulative_equity = test_result.ending_equity

    trades_columns: list[str] = [
        "date",
        "symbol",
        "side",
        "qty",
        "fill_price",
        "slippage_cost",
        "commission_cost",
        "strategy_slug",
    ]

    if not oos_equity_pieces:
        empty_series = pd.Series(dtype=float)
        empty_trades = pd.DataFrame(columns=trades_columns)
        empty_combined = BacktestResult(
            equity_curve=empty_series,
            returns=empty_series,
            positions=pd.DataFrame(),
            trades=empty_trades,
            config=config,
            starting_equity=config.starting_equity,
            ending_equity=config.starting_equity,
        )
        return WalkforwardResult(
            oos_equity_curve=empty_series,
            oos_returns=empty_series,
            oos_trades=empty_trades,
            per_window_params=[],
            combined_result=empty_combined,
        )

    oos_equity = pd.concat(oos_equity_pieces)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index()
    oos_returns = oos_equity.pct_change().fillna(0.0)
    oos_trades = (
        pd.concat(oos_trades_pieces, ignore_index=True)
        if oos_trades_pieces
        else pd.DataFrame(columns=trades_columns)
    )

    combined = BacktestResult(
        equity_curve=oos_equity,
        returns=oos_returns,
        positions=pd.DataFrame(),
        trades=oos_trades,
        config=config,
        starting_equity=config.starting_equity,
        ending_equity=float(oos_equity.iloc[-1]),
        metadata={"walkforward": True, "n_windows": len(per_window_params)},
    )

    return WalkforwardResult(
        oos_equity_curve=oos_equity,
        oos_returns=oos_returns,
        oos_trades=oos_trades,
        per_window_params=per_window_params,
        combined_result=combined,
    )
