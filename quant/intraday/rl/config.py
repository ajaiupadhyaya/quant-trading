"""Configuration for the tabular Q-learning execution agent. No magic numbers per
the Charter; all knobs here. Stylized: tabular RL rediscovers the DP/A-C optimum."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    total_shares: int = 20
    n_steps: int = 10
    n_actions: int = 5
    alpha: float = 0.2
    gamma_discount: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    n_episodes: int = 20_000
    risk_aversion: float = 0.01
    impact_coef_bps: float = 10.0
    adv_dollar: float = 5_000_000_000.0
    sigma: float = 0.02
    dt: float = 1.0
    start_price: float = 100.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.total_shares < 1:
            raise ValueError("total_shares must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.gamma_discount <= 1.0:
            raise ValueError("gamma_discount must be in [0, 1]")
        if not self.epsilon_start >= self.epsilon_end >= 0.0:
            raise ValueError("require epsilon_start >= epsilon_end >= 0")
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")
        for name in ("sigma", "dt", "start_price", "adv_dollar", "impact_coef_bps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
