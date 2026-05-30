"""Backtest output: intraday equity curve + daily-resampled returns so the
existing charter metrics (and, in sub-project C, DSR/PSR/bootstrap) operate on
daily returns exactly as the daily system does."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant.backtest.metrics import max_drawdown, sharpe, total_return
from quant.intraday.sim.fills import Fill


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    impact: float
    spread: float
    financing: float

    @property
    def total(self) -> float:
        return self.commission + self.impact + self.spread + self.financing


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series  # intraday marks (DatetimeIndex)
    fills: list[Fill]
    costs: CostBreakdown
    metadata: dict[str, Any] = field(default_factory=dict)

    def daily_returns(self) -> pd.Series:
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        daily = self.equity_curve.resample("1D").last().dropna()
        return daily.pct_change().dropna()

    def sharpe(self, periods_per_year: int = 252) -> float:
        return sharpe(self.daily_returns(), periods_per_year=periods_per_year)

    def max_drawdown(self) -> float:
        return max_drawdown(self.daily_returns())

    def total_return(self) -> float:
        return total_return(self.daily_returns())
