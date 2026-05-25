"""Hard-coded historical regime windows + per-regime metric breakdown."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant.backtest.metrics import max_drawdown, sharpe, total_return


@dataclass(frozen=True)
class Regime:
    slug: str
    name: str
    start: date
    end: date


REGIMES: tuple[Regime, ...] = (
    Regime("gfc-2008", "2008 Global Financial Crisis", date(2007, 10, 9), date(2009, 3, 9)),
    Regime("china-2015", "2015-16 China Selloff", date(2015, 8, 1), date(2016, 2, 11)),
    Regime("covid-2020", "2020 COVID Crash", date(2020, 2, 19), date(2020, 4, 7)),
    Regime("bear-2022", "2022 Bear Market", date(2022, 1, 3), date(2022, 10, 12)),
    Regime("bull-2024", "2023-24 Recovery Bull", date(2023, 10, 27), date(2024, 12, 31)),
)


@dataclass(frozen=True)
class RegimeBreakdown:
    slug: str
    name: str
    start: date
    end: date
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float


def compute_regime_breakdown(returns: pd.Series) -> list[RegimeBreakdown]:
    """Slice ``returns`` into each regime window and compute key metrics.

    Returns one entry per regime in REGIMES order. Regimes with no overlap
    (including the all-empty case where ``returns`` has no rows) yield zero
    metrics with ``n_days=0``.
    """
    out: list[RegimeBreakdown] = []
    is_dt = isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 0
    for r in REGIMES:
        if is_dt:
            mask = (returns.index >= pd.Timestamp(r.start)) & (returns.index <= pd.Timestamp(r.end))
            slice_ = returns[mask]
        else:
            slice_ = returns.iloc[0:0]
        out.append(
            RegimeBreakdown(
                slug=r.slug,
                name=r.name,
                start=r.start,
                end=r.end,
                n_days=len(slice_),
                total_return=total_return(slice_),
                sharpe=sharpe(slice_),
                max_drawdown=max_drawdown(slice_),
            )
        )
    return out


_MIN_REGIME_DAYS = 30


def count_positive_regimes(breakdown: list[RegimeBreakdown]) -> int:
    """Number of regimes with strictly-positive total return (n_days>0 required)."""
    return sum(1 for b in breakdown if b.n_days > 0 and b.total_return > 0.0)


def count_tested_regimes(breakdown: list[RegimeBreakdown], min_days: int = _MIN_REGIME_DAYS) -> int:
    """How many regimes had enough OOS data to actually be evaluated.

    A regime entirely outside the OOS window (n_days=0) is not a fail — it
    just wasn't tested. Strategies trained on post-2010 data physically cannot
    be evaluated on GFC 2008 or most of the 2015 China selloff. The gate uses
    this to scale the threshold rather than treating unreachable regimes as
    automatic fails.
    """
    return sum(1 for b in breakdown if b.n_days >= min_days)
