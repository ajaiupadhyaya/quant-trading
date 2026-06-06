"""Backtest engine, walk-forward harness, and tear-sheet generator."""

from __future__ import annotations

from quant.backtest.activity import CapacityReport, annualized_turnover, capacity_report
from quant.backtest.combined import CombinedResult, run_combined_book
from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quant.backtest.tearsheet import write_combined_tearsheet, write_tearsheet
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
    "CapacityReport",
    "CombinedResult",
    "WalkforwardResult",
    "WalkforwardWindow",
    "annualized_turnover",
    "capacity_report",
    "iter_windows",
    "run_backtest",
    "run_combined_book",
    "run_walkforward",
    "select_best_params",
    "write_combined_tearsheet",
    "write_tearsheet",
]
