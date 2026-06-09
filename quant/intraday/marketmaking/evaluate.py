"""Sweep risk-aversion gamma to show the A-S tradeoff: low gamma = tight spread, many
fills, higher inventory risk; high gamma = wide spread, fewer fills, controlled
inventory. The market-making analog of the optimal-execution efficient frontier."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.simulator import run_market_making


@dataclass(frozen=True)
class SweepPoint:
    gamma: float
    final_pnl: float
    n_fills: int
    mean_abs_inventory: float
    max_abs_inventory: int
    terminal_inventory: int


def gamma_sweep(prices: list[float], config: MMConfig, gammas: list[float]) -> list[SweepPoint]:
    """Run the simulator on the SAME path+seed for each gamma."""
    pts: list[SweepPoint] = []
    for g in gammas:
        cfg = dataclasses.replace(config, gamma=g)
        r = run_market_making(prices, cfg)
        pts.append(SweepPoint(
            gamma=g,
            final_pnl=r.final_pnl,
            n_fills=r.n_bid_fills + r.n_ask_fills,
            mean_abs_inventory=r.mean_abs_inventory,
            max_abs_inventory=r.max_abs_inventory,
            terminal_inventory=r.terminal_inventory,
        ))
    return pts
