"""GARCH-family volatility forecasters (charter gap #3) — hand-rolled, pure numpy."""

from __future__ import annotations

import math

import numpy as np

from quant.forecast.garch import (
    GarchModel,
    fit_garch,
    garch_conditional_variances,
    garch_forecast_next,
)


def _sim_garch(
    n: int, seed: int, omega: float = 2e-6, alpha: float = 0.08, beta: float = 0.90
) -> np.ndarray:
    """True GARCH(1,1) *return* series (not prices)."""
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - beta)
    r = np.zeros(n)
    for t in range(1, n):
        s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        r[t] = math.sqrt(s2) * rng.standard_normal()
    return r


def _sim_gjr(
    n: int,
    seed: int,
    omega: float = 2e-6,
    alpha: float = 0.03,
    gamma: float = 0.12,
    beta: float = 0.88,
) -> np.ndarray:
    """True GJR-GARCH return series with a strong leverage effect."""
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - beta - 0.5 * gamma)
    r = np.zeros(n)
    for t in range(1, n):
        lev = gamma if r[t - 1] < 0 else 0.0
        s2 = omega + (alpha + lev) * r[t - 1] ** 2 + beta * s2
        r[t] = math.sqrt(s2) * rng.standard_normal()
    return r


def test_fit_garch_recovers_persistent_process() -> None:
    r = _sim_garch(3000, seed=1)
    m = fit_garch(r, kind="garch")
    assert m is not None
    assert m.kind == "garch"
    # Stationary, persistent process: 0 < alpha+beta < 1, beta dominant.
    assert m.alpha >= 0.0
    assert m.beta >= 0.0
    assert 0.5 < m.alpha + m.beta < 1.0
    assert m.gamma == 0.0  # plain GARCH has no leverage term


def test_fit_gjr_has_nonnegative_leverage_term() -> None:
    r = _sim_gjr(3000, seed=2)
    m = fit_garch(r, kind="gjr")
    assert m is not None and m.kind == "gjr"
    assert m.gamma >= 0.0
    # On a leverage-skewed DGP the asymmetry term should be picked up (> 0).
    assert m.gamma > 0.0
    assert m.alpha + m.beta + 0.5 * m.gamma < 1.0


def test_forecast_next_is_positive_finite_floored() -> None:
    r = _sim_garch(2000, seed=3)
    m = fit_garch(r, kind="garch")
    assert m is not None
    f = garch_forecast_next(m, r)
    assert f is not None and math.isfinite(f) and f > 0.0


def test_conditional_variances_shape_and_positivity() -> None:
    r = _sim_garch(1000, seed=4)
    m = fit_garch(r, kind="garch")
    assert m is not None
    s2 = garch_conditional_variances(m, r)
    assert s2.shape == r.shape
    assert np.all(s2 > 0.0) and np.all(np.isfinite(s2))


def test_fit_returns_none_on_short_series() -> None:
    assert fit_garch(np.array([0.01, -0.02, 0.005]), kind="garch") is None
    assert fit_garch(np.array([]), kind="garch") is None


def test_fit_returns_none_on_degenerate_zero_variance() -> None:
    assert fit_garch(np.zeros(500), kind="garch") is None


def test_determinism_identical_inputs_identical_model() -> None:
    r = _sim_garch(1500, seed=7)
    a = fit_garch(r, kind="garch")
    b = fit_garch(r, kind="garch")
    assert a is not None and b is not None
    assert (a.omega, a.alpha, a.beta, a.gamma) == (b.omega, b.alpha, b.beta, b.gamma)


def test_garch_beats_random_walk_on_qlike() -> None:
    """A correct GARCH should forecast GARCH-DGP variance better than RW (squared
    last return) on the proxy-robust QLIKE loss, out-of-sample."""
    from quant.forecast.vol import qlike

    r = _sim_garch(3000, seed=9)
    split = 2000
    train = r[:split]
    m = fit_garch(train, kind="garch")
    assert m is not None
    # one-step-ahead, refit-free, rolling the recursion forward over the holdout
    q_garch: list[float] = []
    q_rw: list[float] = []
    for t in range(split, len(r) - 1):
        f = garch_forecast_next(m, r[: t + 1])
        proxy = r[t + 1] ** 2
        if f is None:
            continue
        q_garch.append(qlike(f, proxy))
        q_rw.append(qlike(max(r[t] ** 2, 1e-8), proxy))
    assert np.mean(q_garch) < np.mean(q_rw)


def test_model_is_frozen_dataclass() -> None:
    r = _sim_garch(800, seed=5)
    m = fit_garch(r, kind="garch")
    assert isinstance(m, GarchModel)
    try:
        m.alpha = 0.5  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("GarchModel must be frozen")
