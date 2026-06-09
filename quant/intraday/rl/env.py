"""Execution MDP for the tabular Q-learning agent. State = (steps_remaining,
inventory_remaining). Action index in [0, n_actions) maps to a sell-fraction of the
remaining inventory. Reward = -(square-root impact + A-C risk penalty); leftover
inventory is force-liquidated at the deadline. The mid follows a seeded ABM path."""

from __future__ import annotations

import math
import random

from quant.backtest.impact import market_impact_bps
from quant.intraday.rl.config import RLConfig

State = tuple[int, int]


def step_cost(
    *,
    traded: int,
    inventory_after: int,
    price: float,
    config: RLConfig,
) -> float:
    """Per-step COST (>= 0): square-root market impact of `traded` shares plus the
    A-C inventory-risk penalty on `inventory_after`. Cost, not reward (reward = -cost)."""
    impact_cost = 0.0
    if traded > 0:
        notional = price * traded
        imp_bps = market_impact_bps(notional, config.adv_dollar, config.impact_coef_bps)
        impact_cost = price * (imp_bps / 1e4) * traded
    risk_cost = config.risk_aversion * config.sigma**2 * inventory_after**2 * config.dt
    return impact_cost + risk_cost


def action_to_shares(action_index: int, inventory: int, n_actions: int) -> int:
    """Map a discrete action to a share count: evenly spaced sell-fractions of the
    remaining inventory, 0 (do nothing) .. 1 (sell all). Clamped to inventory."""
    fraction = action_index / (n_actions - 1)
    return min(inventory, round(fraction * inventory))


class ExecutionEnv:
    def __init__(self, config: RLConfig, *, seed: int) -> None:
        self._cfg = config
        self._rng = random.Random(seed)
        self._steps_remaining = config.n_steps
        self._inventory = config.total_shares
        self._price = config.start_price

    def reset(self) -> State:
        self._steps_remaining = self._cfg.n_steps
        self._inventory = self._cfg.total_shares
        self._price = self._cfg.start_price
        return (self._steps_remaining, self._inventory)

    def step(self, action_index: int) -> tuple[State, float, bool]:
        cfg = self._cfg
        traded = action_to_shares(action_index, self._inventory, cfg.n_actions)
        inventory_after = self._inventory - traded
        cost = step_cost(
            traded=traded,
            inventory_after=inventory_after,
            price=self._price,
            config=cfg,
        )

        self._steps_remaining -= 1
        done = self._steps_remaining == 0

        if done and inventory_after > 0:
            notional = self._price * inventory_after
            imp_bps = market_impact_bps(notional, cfg.adv_dollar, cfg.impact_coef_bps)
            cost += self._price * (imp_bps / 1e4) * inventory_after
            inventory_after = 0

        self._inventory = inventory_after
        self._price = (
            self._price
            + cfg.sigma * math.sqrt(cfg.dt) * self._rng.gauss(0.0, 1.0)
        )
        return (self._steps_remaining, inventory_after), -cost, done
