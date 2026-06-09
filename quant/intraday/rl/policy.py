"""Greedy policy extraction from a trained Q-table, and a noiseless rollout of that
policy into a child-size schedule (for comparison against A-C / TWAP)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import action_to_shares

State = tuple[int, int]


def greedy_action(qtable: NDArray[np.float64], state: State) -> int:
    sr, inv = state
    return int(np.argmax(qtable[sr, inv]))


def rollout_schedule(qtable: NDArray[np.float64], config: RLConfig) -> list[int]:
    """Greedily roll the policy forward into a child-size schedule. Force-liquidates any
    remainder on the final step so the schedule sums to total_shares."""
    cfg = config
    inv = cfg.total_shares
    sched: list[int] = []
    for step in range(cfg.n_steps):
        steps_remaining = cfg.n_steps - step
        action = greedy_action(qtable, (steps_remaining, inv))
        traded = action_to_shares(action, inv, cfg.n_actions)
        if step == cfg.n_steps - 1:
            traded = inv
        sched.append(traded)
        inv -= traded
    return sched
