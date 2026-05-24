"""Backtest engine, walk-forward harness, and tear-sheet generator."""

from __future__ import annotations

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.backtest.tearsheet import write_tearsheet
from quant.backtest.walkforward import (
    WalkforwardResult,
    WalkforwardWindow,
    iter_windows,
    run_walkforward,
    select_best_params,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "WalkforwardResult",
    "WalkforwardWindow",
    "iter_windows",
    "run_backtest",
    "run_walkforward",
    "select_best_params",
    "write_tearsheet",
]
