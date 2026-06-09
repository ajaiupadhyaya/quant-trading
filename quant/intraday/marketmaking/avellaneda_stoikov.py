"""Avellaneda-Stoikov (2008) optimal market-making quotes. Absolute volatility
(price units); the mid follows arithmetic Brownian motion. Pure functions."""

from __future__ import annotations

import math


def reservation_price(
    mid: float, inventory: int, gamma: float, sigma: float, t_remaining: float
) -> float:
    """r = s - q*gamma*sigma^2*(T-t). Long inventory skews the quote center down."""
    return mid - inventory * gamma * sigma * sigma * t_remaining


def optimal_spread(*, gamma: float, sigma: float, t_remaining: float, k: float) -> float:
    """Total bid-ask spread: gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)."""
    return gamma * sigma * sigma * t_remaining + (2.0 / gamma) * math.log(1.0 + gamma / k)


def quotes(
    mid: float, inventory: int, gamma: float, sigma: float, t_remaining: float, k: float
) -> tuple[float, float]:
    """Return (bid, ask) centered on the reservation price, spread wide."""
    r = reservation_price(mid, inventory, gamma, sigma, t_remaining)
    half = optimal_spread(gamma=gamma, sigma=sigma, t_remaining=t_remaining, k=k) / 2.0
    return r - half, r + half
