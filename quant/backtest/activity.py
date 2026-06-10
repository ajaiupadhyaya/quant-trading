"""Trade-activity metrics computed from the backtest trade ledger.

Unlike ``metrics.py`` (every function maps a daily-returns Series -> float),
these take the trade ledger plus the equity curve, because turnover -- and,
later, capacity -- are properties of *trading activity*, not of the returns
stream. That different input shape is why this is a separate module.

Undefined results return 0.0 rather than raising, mirroring the ``metrics.py``
convention so tear-sheet rendering never breaks on edge cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CapacityReport:
    """Strategy capacity: the AUM ceiling implied by liquidity, two ways.

    All participations/capacities are evaluated at the backtest's *current* mean
    equity and assume capital scales fill notionals proportionally (the standard
    capacity assumption). ``capacity_aum`` is the binding (smaller) of the two
    ceilings; ``binding`` names which constraint binds.
    """

    n_fills_total: int
    n_fills_scored: int  # fills with a finite, positive ADV
    median_participation: float  # fraction of dollar-ADV per fill, at current AUM
    p95_participation: float
    max_participation: float
    participation_capacity: float  # AUM at which the p95 fill hits the participation cap
    impact_capacity: float  # AUM at which annual impact drag hits the budget (inf = unbounded)
    capacity_aum: float  # min of the two ceilings (0.0 when undefined)
    binding: str  # "participation" | "impact" | "none"
    max_participation_cap: float  # the cap used
    impact_budget_bps: float  # the annual impact-drag budget used


def capacity_report(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    *,
    max_participation: float = 0.10,
    impact_coef_bps: float = 100.0,
    impact_budget_bps: float = 100.0,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> CapacityReport:
    """Estimate strategy capacity from the trade ledger + a per-fill dollar-ADV.

    Two model-free ceilings, both assuming AUM scales fill notionals linearly:

    * **Participation:** each fill trades ``notional/adv`` of a day's volume.
      Scaling AUM by ``k`` scales participation by ``k``; the largest ``k`` that
      keeps the *95th-percentile* fill at or under ``max_participation`` gives
      ``participation_capacity = mean_equity * max_participation / p95``. The p95
      (not the max) is the headline so one lone illiquid fill doesn't dominate —
      ``max_participation`` (the field) exposes the worst case.
    * **Impact:** square-root impact cost per fill is
      ``notional * impact_coef_bps*sqrt(notional/adv)/1e4``; summed and annualized
      it is a fractional drag ``g0`` on equity that grows as ``sqrt(k)``. The AUM
      at which that drag reaches ``impact_budget_bps`` is
      ``impact_capacity = mean_equity * (budget/g0)^2`` (``inf`` if impact is off
      or zero — it never binds).

    Reads ``qty``, ``fill_price`` and ``adv_dollar`` from the ledger; a ledger
    without ``adv_dollar`` (legacy runs) scores zero fills and returns
    ``binding="none"`` rather than raising. Research/reporting only — drives
    nothing.
    """
    none = CapacityReport(
        n_fills_total=0 if trades is None else len(trades),
        n_fills_scored=0,
        median_participation=0.0,
        p95_participation=0.0,
        max_participation=0.0,
        participation_capacity=0.0,
        impact_capacity=0.0,
        capacity_aum=0.0,
        binding="none",
        max_participation_cap=max_participation,
        impact_budget_bps=impact_budget_bps,
    )
    if trades is None or len(trades) == 0 or "adv_dollar" not in trades.columns:
        return none
    n_days = len(equity_curve)
    if n_days == 0:
        return none
    mean_equity = float(equity_curve.mean())
    if not np.isfinite(mean_equity) or mean_equity <= _EQUITY_EPS:
        return none

    notional = (trades["qty"].abs() * trades["fill_price"]).to_numpy(dtype=float)
    adv = trades["adv_dollar"].to_numpy(dtype=float)
    ok = np.isfinite(notional) & np.isfinite(adv) & (notional > 0.0) & (adv > 0.0)
    notional, adv = notional[ok], adv[ok]
    if notional.size == 0:
        return none

    participation = notional / adv
    p_med = float(np.median(participation))
    p95 = float(np.percentile(participation, 95))
    p_max = float(participation.max())

    participation_capacity = mean_equity * (max_participation / p95) if p95 > 0.0 else math.inf

    # Impact ceiling: annualized fractional drag at current AUM, growing as sqrt(k).
    impact_cost = notional * (impact_coef_bps * np.sqrt(participation)) / 1e4
    annual_impact = float(impact_cost.sum()) * (periods_per_year / n_days)
    g0 = annual_impact / mean_equity
    budget = impact_budget_bps / 1e4
    impact_capacity = mean_equity * (budget / g0) ** 2 if g0 > 0.0 else math.inf

    capacity_aum = min(participation_capacity, impact_capacity)
    if not np.isfinite(capacity_aum):
        binding = "none"
        capacity_aum = 0.0
    elif impact_capacity < participation_capacity:
        binding = "impact"
    else:
        binding = "participation"

    return CapacityReport(
        n_fills_total=len(trades),
        n_fills_scored=int(notional.size),
        median_participation=p_med,
        p95_participation=p95,
        max_participation=p_max,
        participation_capacity=participation_capacity,
        impact_capacity=impact_capacity,
        capacity_aum=capacity_aum,
        binding=binding,
        max_participation_cap=max_participation,
        impact_budget_bps=impact_budget_bps,
    )
