"""Performance metrics on a daily-returns Series.

All functions take a pd.Series of daily simple returns and return a float.
By convention, undefined results (empty input, zero vol, no wins) return 0.0
rather than NaN, so tear-sheet rendering never breaks on edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    arr = returns.to_numpy(dtype=float)
    return float(np.prod(1.0 + arr) - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    tr = total_return(returns)
    years = len(returns) / periods_per_year
    if years <= 0 or tr <= -1.0:
        return 0.0
    return float((1.0 + tr) ** (1.0 / years) - 1.0)


_STD_EPS = 1e-12


def sharpe(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if std <= _STD_EPS:
        return 0.0
    mean = float(returns.mean())
    return float(mean / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    dd_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    if dd_std <= _STD_EPS:
        return 0.0
    mean = float(returns.mean())
    return float(mean / dd_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown as a negative number."""
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def win_rate(returns: pd.Series) -> float:
    """Fraction of strictly-positive returns among non-zero returns."""
    if len(returns) == 0:
        return 0.0
    nonzero = returns[returns != 0.0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())
