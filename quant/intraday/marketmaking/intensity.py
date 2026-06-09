"""Poisson fill-intensity model: a quote at distance `delta` from mid fills with
intensity lambda(delta) = a*exp(-k*delta). Over a step dt the fill probability is
1 - exp(-lambda*dt), clamped to [0, 1].

Parameter names follow the A-S literature (Avellaneda-Stoikov 2008) but use
lowercase `a` to match the ``fill_rate_a`` field in MMConfig.
"""

from __future__ import annotations

import math
import random

# math.exp overflows above ~709.78; clamp the exponent defensively.
_MAX_EXP: float = 709.0


def fill_intensity(*, delta: float, a: float, k: float) -> float:
    """lambda(delta) = a * exp(-k * delta). Larger distance -> lower intensity."""
    exponent = min(-k * delta, _MAX_EXP)
    return a * math.exp(exponent)


def fill_probability(*, delta: float, a: float, k: float, dt: float) -> float:
    """P(>=1 fill in dt) = 1 - exp(-lambda*dt), clamped to [0, 1]."""
    lam = fill_intensity(delta=delta, a=a, k=k)
    p = 1.0 - math.exp(-lam * dt)
    return min(1.0, max(0.0, p))


def draws_fill(prob: float, rng: random.Random) -> bool:
    """Bernoulli draw against `prob` using the supplied seeded RNG."""
    return rng.random() < prob
