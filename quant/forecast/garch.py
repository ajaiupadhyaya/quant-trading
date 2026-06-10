"""Deterministic GARCH-family volatility forecasters (pure numpy, no dependencies).

Hand-rolled to match the codebase's model ethos (HAR/EWMA/HMM/Kalman/ridge/GBM are
all from-scratch numpy): fully deterministic given ``(returns, kind)``, transparent,
PIT-stable, and dependency-free (no ``arch``, no scipy). Closes charter gap #3
(ARIMA/GARCH time-series modeling) for the *variance* process, which is what the
charter wants the gap to feed (vol-targeting / sizing).

Two members of the family:

* **GARCH(1,1)** (Bollerslev 1986): ``s2_t = omega + alpha*eps2_{t-1} + beta*s2_{t-1}``
* **GJR-GARCH(1,1,1)** (Glosten-Jagannathan-Runkle 1993): adds a leverage term
  ``gamma*1[eps_{t-1}<0]*eps2_{t-1}`` — down-moves raise volatility more than
  up-moves, the empirically strong equity-index asymmetry plain GARCH misses.

Both are fit by Gaussian **QMLE** under **variance targeting** (the unconditional
variance is pinned to the sample variance of demeaned returns, leaving only the
persistence parameters to optimize) via a deterministic coarse-grid → zoom-refine
search over the stationarity simplex — no RNG, no external optimizer.

Like HAR before promotion, these are evaluated the only honest way — walk-forward,
one-day-ahead, QLIKE-primary, with a Diebold-Mariano significance test (see
:func:`quant.forecast.vol.walk_forward_eval`) — and drive nothing until they earn
it. This module is wired to no sizing, tilt, or order path.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

_VAR_FLOOR = 1e-12  # variance floor — guards log/division in the QMLE recursion
_MIN_OBS = 250  # too little to fit a persistence structure honestly
_MAX_PERSIST = 0.999  # stationarity ceiling on alpha + beta (+ gamma/2)

GarchKind = Literal["garch", "gjr"]


@dataclass(frozen=True)
class GarchModel:
    """Fitted GARCH-family parameters under variance targeting.

    ``omega`` is implied by the targeted unconditional variance ``var_uncond`` and
    the persistence parameters; ``gamma`` is the GJR leverage term (0 for plain
    GARCH). ``mean`` is the sample mean removed from returns before fitting.
    """

    kind: GarchKind
    omega: float
    alpha: float
    beta: float
    gamma: float
    var_uncond: float
    mean: float
    n_obs: int


def _omega(kind: GarchKind, alpha: float, beta: float, gamma: float, var_uncond: float) -> float:
    """Variance-targeting intercept: pins the unconditional variance to ``var_uncond``."""
    persist = alpha + beta + (0.5 * gamma if kind == "gjr" else 0.0)
    return float(max(var_uncond * (1.0 - persist), _VAR_FLOOR))


def _recursion(
    eps: np.ndarray,
    kind: GarchKind,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    var_uncond: float,
) -> np.ndarray:
    """Conditional-variance path ``s2[t]`` for demeaned innovations ``eps`` (seeded
    with the unconditional variance). ``s2[t]`` is the variance *of* ``eps[t]``
    given information through ``t-1`` — i.e. a one-step-ahead forecast each step.
    """
    n = eps.size
    s2 = np.empty(n, dtype=float)
    prev = var_uncond
    s2[0] = prev
    for t in range(1, n):
        e2 = eps[t - 1] * eps[t - 1]
        lev = gamma if (kind == "gjr" and eps[t - 1] < 0.0) else 0.0
        prev = omega + (alpha + lev) * e2 + beta * prev
        if prev < _VAR_FLOOR:
            prev = _VAR_FLOOR
        s2[t] = prev
    return s2


def _neg_loglik(
    eps: np.ndarray,
    kind: GarchKind,
    alpha: float,
    beta: float,
    gamma: float,
    var_uncond: float,
) -> float:
    """Gaussian negative log-likelihood (up to a constant) for given parameters."""
    omega = _omega(kind, alpha, beta, gamma, var_uncond)
    s2 = _recursion(eps, kind, omega, alpha, beta, gamma, var_uncond)
    # 0.5 * sum(log s2 + eps^2 / s2); drop the constant 0.5*log(2pi)*n.
    return float(0.5 * np.sum(np.log(s2) + (eps * eps) / s2))


def _candidates(
    kind: GarchKind, center: tuple[float, float, float], radius: float, steps: int
) -> Iterator[tuple[float, float, float]]:
    """Yield (alpha, beta, gamma) grid points in a box around ``center`` that
    satisfy the non-negativity + stationarity constraints. Deterministic order."""
    a0, b0, g0 = center
    axis = np.linspace(-radius, radius, steps)
    gammas = axis if kind == "gjr" else np.array([0.0])
    for da in axis:
        a = a0 + da
        if a < 0.0:
            continue
        for db in axis:
            b = b0 + db
            if b < 0.0:
                continue
            for dg in gammas:
                g = g0 + dg if kind == "gjr" else 0.0
                if g < 0.0:
                    continue
                if a + b + 0.5 * g >= _MAX_PERSIST:
                    continue
                yield a, b, g


def fit_garch(returns: np.ndarray, *, kind: GarchKind = "garch") -> GarchModel | None:
    """Fit a GARCH(1,1) or GJR-GARCH(1,1,1) by variance-targeted Gaussian QMLE.

    Deterministic coarse-grid → zoom-refine over the stationarity simplex. Returns
    ``None`` on too-few or degenerate (zero-variance) data so callers fail soft.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < _MIN_OBS:
        return None
    mean = float(r.mean())
    eps = r - mean
    var_uncond = float(np.mean(eps * eps))
    if var_uncond <= _VAR_FLOOR:
        return None

    # Coarse grid centered on a typical high-persistence equity regime, then two
    # zoom-refine passes around the running best. Fully deterministic.
    best = (0.05, 0.90, 0.0)
    best_nll = _neg_loglik(eps, kind, *best, var_uncond)
    for da in np.linspace(0.0, 0.30, 13):
        for db in np.linspace(0.50, 0.98, 13):
            for dg in np.linspace(0.0, 0.30, 7) if kind == "gjr" else (0.0,):
                a, b, g = float(da), float(db), float(dg)
                if a + b + 0.5 * g >= _MAX_PERSIST:
                    continue
                nll = _neg_loglik(eps, kind, a, b, g, var_uncond)
                if nll < best_nll:
                    best_nll, best = nll, (a, b, g)

    radius, steps = 0.05, 7
    for _ in range(2):
        improved = False
        for a, b, g in _candidates(kind, best, radius, steps):
            nll = _neg_loglik(eps, kind, a, b, g, var_uncond)
            if nll < best_nll - 1e-12:
                best_nll, best, improved = nll, (a, b, g), True
        radius *= 0.4
        if not improved:
            radius *= 0.5

    alpha, beta, gamma = best
    return GarchModel(
        kind=kind,
        omega=_omega(kind, alpha, beta, gamma, var_uncond),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        var_uncond=var_uncond,
        mean=mean,
        n_obs=int(r.size),
    )


def garch_conditional_variances(model: GarchModel, returns: np.ndarray) -> np.ndarray:
    """Full conditional-variance path for ``returns`` under a fitted model."""
    r = np.asarray(returns, dtype=float)
    eps = r - model.mean
    if eps.size == 0:
        return np.array([])
    return _recursion(
        eps, model.kind, model.omega, model.alpha, model.beta, model.gamma, model.var_uncond
    )


def garch_forecast_next(model: GarchModel, returns: np.ndarray) -> float | None:
    """One-step-ahead variance forecast ``s2_{T+1}`` from returns through ``T``."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    eps = r - model.mean
    s2 = _recursion(
        eps, model.kind, model.omega, model.alpha, model.beta, model.gamma, model.var_uncond
    )
    e2 = float(eps[-1] * eps[-1])
    lev = model.gamma if (model.kind == "gjr" and eps[-1] < 0.0) else 0.0
    forecast = model.omega + (model.alpha + lev) * e2 + model.beta * float(s2[-1])
    return float(max(forecast, _VAR_FLOOR)) if math.isfinite(forecast) else None
