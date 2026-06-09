"""Configuration for the Avellaneda-Stoikov market-making simulator. No magic
numbers per the Charter; all knobs live here. This is a STYLIZED model: A and k are
assumed intensity parameters, not fit to live fills."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MMConfig:
    gamma: float = 0.1          # inventory risk aversion
    k: float = 1.5              # order-book depth / intensity decay
    fill_rate_a: float = 1.5    # base fill intensity; scaled for absolute-price quote
                                # distances (delta ~ spread/2 ~ O(1)) so fills are
                                # PROBABILISTIC, not saturated. (A large A like 140
                                # saturates fill prob to 1 every tick and hides the
                                # inventory-risk tradeoff.)
    horizon_seconds: float = 600.0  # T (one 10-min episode)
    dt_seconds: float = 1.0     # simulation step
    sigma: float = 0.02         # ABSOLUTE vol, price units per sqrt(second)
    lot_size: int = 1           # shares per fill
    seed: int = 7

    def __post_init__(self) -> None:
        for name in ("gamma", "k", "fill_rate_a", "horizon_seconds", "dt_seconds", "sigma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.lot_size < 1:
            raise ValueError("lot_size must be >= 1")

    @property
    def n_steps(self) -> int:
        return int(self.horizon_seconds / self.dt_seconds)
