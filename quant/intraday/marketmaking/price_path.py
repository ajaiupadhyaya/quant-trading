"""Arithmetic Brownian motion mid-price path (Avellaneda-Stoikov uses absolute vol):
s_{t+1} = s_t + sigma*sqrt(dt)*z, z~N(0,1). Seeded for reproducibility."""

from __future__ import annotations

import math
import random


def abm_path(*, s0: float, sigma: float, dt: float, n_steps: int, rng: random.Random) -> list[float]:
    """Return [s_0, s_1, ..., s_n] under arithmetic Brownian motion."""
    step_sd = sigma * math.sqrt(dt)
    path = [s0]
    s = s0
    for _ in range(n_steps):
        s = s + step_sd * rng.gauss(0.0, 1.0)
        path.append(s)
    return path
