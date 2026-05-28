"""Trade-activity metrics computed from the backtest trade ledger.

Unlike ``metrics.py`` (every function maps a daily-returns Series -> float),
these take the trade ledger plus the equity curve, because turnover -- and,
later, capacity -- are properties of *trading activity*, not of the returns
stream. That different input shape is why this is a separate module.

Undefined results return 0.0 rather than raising, mirroring the ``metrics.py``
convention so tear-sheet rendering never breaks on edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_EQUITY_EPS = 1e-9


def annualized_turnover(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """One-way, annualized portfolio turnover from the trade ledger.

    ``traded_notional = sum(|qty| * fill_price)`` over every fill;
    ``one_way = traded_notional / 2`` so a full round-trip reads as 100%;
    ``annualized = (one_way / mean_equity) * (periods_per_year / n_days)``.

    Uses actual fills (including slipped fill prices and zero-crossing
    flatten-and-reopen), not an idealized weight diff. ``trades`` must expose
    ``qty`` and ``fill_price`` columns (a ``BacktestResult.trades`` frame).
    Returns 0.0 when undefined (empty ledger, empty/zero-mean equity).
    """
    if trades is None or len(trades) == 0:
        return 0.0
    n_days = len(equity_curve)
    if n_days == 0:
        return 0.0
    mean_equity = float(equity_curve.mean())
    if not np.isfinite(mean_equity) or mean_equity <= _EQUITY_EPS:
        return 0.0
    notional = trades["qty"].abs() * trades["fill_price"]
    traded_notional = float(notional.sum())
    if notional.isna().any() or not np.isfinite(traded_notional):
        return 0.0
    one_way = traded_notional / 2.0
    return float((one_way / mean_equity) * (periods_per_year / n_days))
