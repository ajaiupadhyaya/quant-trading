"""Evaluate the learned policy against the Almgren-Chriss optimum and TWAP, costing all
three schedules through ONE shared env cost model (step_cost) over seeded ABM paths."""

from __future__ import annotations

import math
import random
from typing import Any

from quant.intraday.execution.almgren_chriss import optimal_schedule
from quant.intraday.execution.baselines import twap
from quant.intraday.execution.calibrate import calibrate
from quant.intraday.execution.config import ExecConfig
from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import step_cost
from quant.intraday.rl.policy import rollout_schedule
from quant.intraday.rl.qlearning import train


def cost_schedule(schedule: list[int], config: RLConfig, *, seed: int) -> float:
    """Replay a fixed child-size schedule through the env cost model on one seeded ABM
    path; force-liquidate any remainder on the last step. Returns total COST (>= 0)."""
    cfg = config
    rng = random.Random(seed)
    price = cfg.start_price
    inv = cfg.total_shares
    total = 0.0
    n = len(schedule)
    for i, raw in enumerate(schedule):
        traded = min(inv, raw)
        if i == n - 1:
            traded = inv
        inv_after = inv - traded
        total += step_cost(traded=traded, inventory_after=inv_after, price=price, config=cfg)
        inv = inv_after
        price = price + cfg.sigma * math.sqrt(cfg.dt) * rng.gauss(0.0, 1.0)
    return total


def _ac_schedule(config: RLConfig) -> list[int]:
    """Almgren-Chriss optimal child sizes using the SAME risk aversion as the RL reward;
    eta calibrated from the same sqrt-impact model (local linearization, as in A)."""
    cfg = config
    per_slice = max(1, cfg.total_shares // cfg.n_steps)
    _, eta, gamma = calibrate(
        price=cfg.start_price,
        slice_shares=per_slice,
        adv_dollar=cfg.adv_dollar,
        recent_returns=[0.0],
        config=ExecConfig(impact_coef_bps=cfg.impact_coef_bps),
    )
    plan = optimal_schedule(
        total_shares=cfg.total_shares,
        n_intervals=cfg.n_steps,
        tau=cfg.dt,
        sigma=cfg.sigma,
        eta=eta,
        gamma=gamma,
        risk_aversion=cfg.risk_aversion,
    )
    return plan.child_sizes


def _mean_cost(schedule: list[int], config: RLConfig, n_eval_paths: int) -> float:
    return (
        sum(
            cost_schedule(schedule, config, seed=config.seed + 10_000 + j)
            for j in range(n_eval_paths)
        )
        / n_eval_paths
    )


def compare(config: RLConfig, n_eval_paths: int = 200) -> dict[str, Any]:
    """Train the agent and compare mean execution cost: learned vs A-C vs TWAP."""
    result = train(config)
    learned = rollout_schedule(result.qtable, config)
    ac = _ac_schedule(config)
    tw = twap(total_shares=config.total_shares, n_intervals=config.n_steps)
    return {
        "learned": _mean_cost(learned, config, n_eval_paths),
        "almgren_chriss": _mean_cost(ac, config, n_eval_paths),
        "twap": _mean_cost(tw, config, n_eval_paths),
        "learned_schedule": learned,
        "training_curve": result.training_curve,
    }
