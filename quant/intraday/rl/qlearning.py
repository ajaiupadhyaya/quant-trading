"""Tabular Q-learning for the execution MDP. Epsilon-greedy with a linear decay;
deterministic given config.seed. Returns the Q-table and a per-block mean-episode-cost
training curve (to show convergence)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import ExecutionEnv


@dataclass(frozen=True)
class TrainResult:
    qtable: NDArray[np.float64]
    training_curve: list[float]


def _epsilon(episode: int, cfg: RLConfig) -> float:
    frac = episode / max(1, cfg.n_episodes - 1)
    return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac


def train(config: RLConfig) -> TrainResult:
    cfg = config
    q: NDArray[np.float64] = np.zeros(
        (cfg.n_steps + 1, cfg.total_shares + 1, cfg.n_actions), dtype=np.float64
    )
    explore_rng = random.Random(cfg.seed)
    n_blocks = 20
    block_size = max(1, cfg.n_episodes // n_blocks)
    curve: list[float] = []
    block_cost = 0.0
    block_count = 0

    for ep in range(cfg.n_episodes):
        env = ExecutionEnv(cfg, seed=cfg.seed + 1 + ep)
        sr, inv = env.reset()
        eps = _epsilon(ep, cfg)
        ep_cost = 0.0
        done = False
        while not done:
            if explore_rng.random() < eps:
                action = explore_rng.randrange(cfg.n_actions)
            else:
                action = int(np.argmax(q[sr, inv]))
            (nsr, ninv), reward, done = env.step(action)
            ep_cost += -reward
            future = 0.0 if done else float(np.max(q[nsr, ninv]))
            target = reward + cfg.gamma_discount * future
            q[sr, inv, action] += cfg.alpha * (target - q[sr, inv, action])
            sr, inv = nsr, ninv
        block_cost += ep_cost
        block_count += 1
        if block_count == block_size:
            curve.append(block_cost / block_count)
            block_cost = 0.0
            block_count = 0
    if block_count:
        curve.append(block_cost / block_count)
    return TrainResult(qtable=q, training_curve=curve)
