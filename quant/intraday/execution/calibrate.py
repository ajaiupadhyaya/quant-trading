"""Calibrate Almgren-Chriss inputs (sigma, eta, gamma) for a parent order.

eta (linear temporary impact) is a local linearization of the repo's SQRT impact
model (quant.backtest.impact.market_impact_bps) at the planned per-slice size, so the
A-C closed form stays usable while being anchored to the real impact curve. The sim
evaluation still uses the true sqrt model, so the closed-form-vs-realized gap is
visible (spec section 6)."""

from __future__ import annotations

import statistics

from quant.backtest.impact import market_impact_bps
from quant.intraday.execution.config import ExecConfig


def calibrate(
    *,
    price: float,
    slice_shares: int,
    adv_dollar: float,
    recent_returns: list[float],
    config: ExecConfig,
) -> tuple[float, float, float]:
    """Return (sigma, eta, gamma).

    sigma: realized stdev of recent returns, in PRICE units (return-stdev * price).
    eta:   per-share temporary-impact slope ($ per share, per share traded), from a
           local linearization of the sqrt model at `slice_shares`.
    gamma: permanent-impact coef = perm_impact_frac * eta.
    """
    ret_sd = statistics.pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
    sigma = ret_sd * price

    slice_shares = max(1, slice_shares)
    slice_notional = price * slice_shares
    impact_bps = market_impact_bps(slice_notional, adv_dollar, config.impact_coef_bps)
    per_share_cost = price * (impact_bps * 1e-4)
    eta = per_share_cost / slice_shares
    if eta <= 0.0:
        eta = 1e-9
    gamma = config.perm_impact_frac * eta
    return sigma, eta, gamma
