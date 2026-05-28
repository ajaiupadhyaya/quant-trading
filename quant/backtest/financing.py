"""Daily borrow + margin-financing costs, computed from carried positions.

Unlike ``engine.apply_costs`` (a per-fill transaction cost), this is a per-day
holding cost on the positions and cash carried overnight. It takes plain rate
floats rather than a ``BacktestConfig`` so it stays standalone and testable and
so ``engine.py`` can import it without a circular dependency.

Costs only — no interest credits (no short rebate, no idle-cash interest).
Undefined / degenerate inputs yield 0.0 components; the function never raises,
mirroring the engine's tolerance for sparse bars.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class FinancingCharge:
    """Breakdown of one bar's financing cost, in dollars."""

    borrow_cost: float
    margin_financing_cost: float

    @property
    def total(self) -> float:
        return self.borrow_cost + self.margin_financing_cost


def financing_charge(
    positions: Mapping[str, int],
    prior_close: Mapping[str, float],
    cash: float,
    days_elapsed: int,
    annual_borrow_bps: float,
    annual_financing_bps: float,
) -> FinancingCharge:
    """Borrow fee on short notional + financing on a margin debit, actual/365.

    ``positions``/``prior_close`` are the holdings carried overnight and the
    PRIOR bar's close prices (no lookahead). A short whose price is missing or
    non-finite contributes 0. ``days_elapsed`` is calendar days since the prior
    bar; ``<= 0`` yields a zero charge.
    """
    if days_elapsed <= 0:
        return FinancingCharge(0.0, 0.0)
    year_frac = days_elapsed / _DAYS_PER_YEAR

    short_notional = 0.0
    for sym, qty in positions.items():
        if qty >= 0:
            continue
        price = prior_close.get(sym)
        if price is None or not math.isfinite(price):
            continue
        short_notional += abs(qty) * price

    borrow_cost = short_notional * (annual_borrow_bps / 1e4) * year_frac
    margin_debit = max(0.0, -cash)
    margin_financing_cost = margin_debit * (annual_financing_bps / 1e4) * year_frac
    return FinancingCharge(borrow_cost=borrow_cost, margin_financing_cost=margin_financing_cost)
