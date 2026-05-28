"""Black-Scholes-Merton pricing + Greeks + implied vol.

Pure functions, no I/O, no state. Continuous dividend yield ``q``. Every
function degrades to ``nan`` on non-finite / non-positive inputs (callers
guard) and never raises on finite inputs -- matching quant/sizing/components.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Greeks:
    """First-order Greeks (theta/rho per year, vega per 1.00 vol point)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _valid(spot: float, strike: float, vol: float, t_years: float) -> bool:
    return (
        math.isfinite(spot)
        and math.isfinite(strike)
        and math.isfinite(vol)
        and math.isfinite(t_years)
        and spot > 0.0
        and strike > 0.0
        and vol > 0.0
    )


def _d1_d2(
    spot: float, strike: float, t_years: float, vol: float, r: float, q: float
) -> tuple[float, float]:
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def bs_price(
    spot: float, strike: float, t_years: float, vol: float, r: float, q: float, right: str
) -> float:
    """Black-Scholes-Merton price. ``right`` in {"call","put"}."""
    if t_years <= 0.0:
        if right == "call":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)
    if not _valid(spot, strike, vol, t_years):
        return float("nan")
    d1, d2 = _d1_d2(spot, strike, t_years, vol, r, q)
    disc_q = math.exp(-q * t_years)
    disc_r = math.exp(-r * t_years)
    if right == "call":
        return float(spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2))
    return float(strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1))


def bs_greeks(
    spot: float, strike: float, t_years: float, vol: float, r: float, q: float, right: str
) -> Greeks:
    """Delta, gamma, vega, theta (per year), rho. nan-filled on bad input."""
    if not _valid(spot, strike, vol, t_years) or t_years <= 0.0:
        nan = float("nan")
        return Greeks(nan, nan, nan, nan, nan)
    d1, d2 = _d1_d2(spot, strike, t_years, vol, r, q)
    sqrt_t = math.sqrt(t_years)
    disc_q = math.exp(-q * t_years)
    disc_r = math.exp(-r * t_years)
    pdf_d1 = float(norm.pdf(d1))
    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    if right == "call":
        delta = disc_q * float(norm.cdf(d1))
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
            - r * strike * disc_r * float(norm.cdf(d2))
            + q * spot * disc_q * float(norm.cdf(d1))
        )
        rho = strike * t_years * disc_r * float(norm.cdf(d2))
    else:
        delta = -disc_q * float(norm.cdf(-d1))
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
            + r * strike * disc_r * float(norm.cdf(-d2))
            - q * spot * disc_q * float(norm.cdf(-d1))
        )
        rho = -strike * t_years * disc_r * float(norm.cdf(-d2))
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    price: float, spot: float, strike: float, t_years: float, r: float, q: float, right: str
) -> float:
    """Brent-solve implied vol on [1e-4, 5.0]; nan if price is unreachable."""
    if not (math.isfinite(price) and price > 0.0) or t_years <= 0.0 or spot <= 0.0:
        return float("nan")

    def objective(vol: float) -> float:
        return bs_price(spot, strike, t_years, vol, r, q, right) - price

    try:
        lo, hi = objective(1e-4), objective(5.0)
        if lo * hi > 0.0:  # not bracketed -> price outside model range
            return float("nan")
        return float(brentq(objective, 1e-4, 5.0, maxiter=100, xtol=1e-8))
    except (ValueError, RuntimeError):
        return float("nan")
