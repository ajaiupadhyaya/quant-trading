"""Deterministic Avellaneda-Stoikov market-making simulator. Steps a mid-price path,
quotes via the A-S model, draws fills via the intensity model, tracks inventory/cash/
P&L. Fully determined by (prices, config) including config.seed."""

from __future__ import annotations

import random
from dataclasses import dataclass

from quant.intraday.marketmaking.avellaneda_stoikov import quotes
from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.intensity import draws_fill, fill_probability


@dataclass(frozen=True)
class MMResult:
    final_pnl: float
    cash: float
    n_bid_fills: int
    n_ask_fills: int
    inventory_path: list[int]
    mean_abs_inventory: float
    max_abs_inventory: int
    terminal_inventory: int
    spread_captured: float


def run_market_making(prices: list[float], config: MMConfig) -> MMResult:
    rng = random.Random(config.seed)
    inventory = 0
    cash = 0.0
    spread_captured = 0.0
    n_bid = 0
    n_ask = 0
    inv_path = [0]
    horizon = config.horizon_seconds
    dt = config.dt_seconds
    lot = config.lot_size

    # The last price has no "next step" to fill against; quote on prices[:-1] and
    # mark the book at prices[-1] at the end.
    for i, mid in enumerate(prices[:-1]):
        tau = max(0.0, horizon - i * dt)
        bid, ask = quotes(mid, inventory, config.gamma, config.sigma, tau, config.k)
        p_bid = fill_probability(delta=mid - bid, a=config.fill_rate_a, k=config.k, dt=dt)
        p_ask = fill_probability(delta=ask - mid, a=config.fill_rate_a, k=config.k, dt=dt)
        if draws_fill(p_bid, rng):           # we BUY at our bid
            inventory += lot
            cash -= bid * lot
            spread_captured += abs(mid - bid) * lot
            n_bid += 1
        if draws_fill(p_ask, rng):           # we SELL at our ask
            inventory -= lot
            cash += ask * lot
            spread_captured += abs(ask - mid) * lot
            n_ask += 1
        inv_path.append(inventory)

    last_mid = prices[-1]
    final_pnl = cash + inventory * last_mid
    abs_inv = [abs(q) for q in inv_path]
    return MMResult(
        final_pnl=final_pnl,
        cash=cash,
        n_bid_fills=n_bid,
        n_ask_fills=n_ask,
        inventory_path=inv_path,
        mean_abs_inventory=sum(abs_inv) / len(abs_inv),
        max_abs_inventory=max(abs_inv),
        terminal_inventory=inventory,
        spread_captured=spread_captured,
    )
