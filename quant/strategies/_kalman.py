"""Kalman filter for time-varying hedge ratios.

Elliott-van der Hoek-Malcolm (2005) state-space form for cointegration:

  Observation: log_a_t = beta_t * log_b_t + alpha_t + eps_t,  eps_t ~ N(0, R)
  State:       [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + eta_t, eta_t ~ N(0, Q)

In other words: the hedge ratio is a 2-D random walk (beta + intercept) observed
through a noisy linear regression on log prices. This implementation runs the
standard Kalman recursion in numpy with a diagonal process-noise covariance
``Q = diag(delta, delta)`` and a scalar observation noise ``R``. Hyper-parameters
delta and R are exposed so the strategy can tune responsiveness vs. smoothness.

Returns the final (beta, alpha) plus the residuals so downstream code can
compute spread z-scores the same way OLS-fit pairs do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanHedgeFit:
    """Final state + residuals from a Kalman hedge run."""

    beta: float
    alpha: float
    residuals: np.ndarray
    spread_std: float


def kalman_hedge(
    log_a: np.ndarray,
    log_b: np.ndarray,
    delta: float = 1e-5,
    obs_var: float = 1e-3,
) -> KalmanHedgeFit | None:
    """Run the Kalman recursion and return the final hedge + residual series.

    Args:
        log_a, log_b: aligned log-price arrays (same length, ≥ 30 obs).
        delta: process-noise variance for each state component. Smaller =
            smoother hedge ratio (closer to OLS); larger = more responsive.
        obs_var: observation noise variance R. Higher dampens the gain.

    Returns ``None`` on degenerate input (too short / mismatched lengths).
    """
    a = np.asarray(log_a, dtype=float)
    b = np.asarray(log_b, dtype=float)
    if a.shape != b.shape or a.size < 30:
        return None

    # Initial state: OLS over the full window. This is a Bayesian prior centered
    # on the static-hedge estimate — the filter then walks beta/alpha away from
    # it as new evidence arrives.
    x_mean = float(b.mean())
    y_mean = float(a.mean())
    cov_xy = float(((b - x_mean) * (a - y_mean)).sum())
    var_x = float(((b - x_mean) ** 2).sum())
    if var_x <= 0:
        return None
    beta0 = cov_xy / var_x
    alpha0 = y_mean - beta0 * x_mean

    state = np.array([beta0, alpha0], dtype=float)
    cov = np.eye(2) * 1.0  # diffuse prior
    q = np.eye(2) * float(delta)
    r = float(obs_var)

    residuals = np.zeros_like(a)
    for t in range(a.size):
        h = np.array([b[t], 1.0])
        # Predict
        pred_state = state.copy()
        pred_cov = cov + q
        # Innovation
        y_hat = float(h @ pred_state)
        innov = a[t] - y_hat
        s = float(h @ pred_cov @ h) + r
        if s <= 0:
            return None
        k = (pred_cov @ h) / s
        # Update
        state = pred_state + k * innov
        cov = (np.eye(2) - np.outer(k, h)) @ pred_cov
        residuals[t] = innov

    spread_std = float(residuals.std(ddof=1))
    if spread_std <= 1e-9 or not np.isfinite(spread_std):
        return None

    return KalmanHedgeFit(
        beta=float(state[0]),
        alpha=float(state[1]),
        residuals=residuals,
        spread_std=spread_std,
    )
