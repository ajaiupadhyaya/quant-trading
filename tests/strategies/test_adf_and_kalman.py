"""Tests for ADF + Kalman extensions to the pairs strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies._kalman import kalman_hedge
from quant.strategies._pairs_discovery import (
    _EG_CV_5PCT,
    engle_granger_adf_stat,
    fit_pair,
)


def _stationary_series(n: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + rng.normal(0, 0.01)
    return eps


def test_adf_rejects_unit_root_on_stationary_series() -> None:
    # Fast mean reversion → ADF should reject (stat well below the 5% CV)
    eps = _stationary_series(n=400, rho=0.5, seed=0)
    stat = engle_granger_adf_stat(eps, max_lag=1)
    assert stat < _EG_CV_5PCT, f"expected stat < {_EG_CV_5PCT}, got {stat}"


def test_adf_does_not_reject_random_walk() -> None:
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0, 0.01, 400))
    stat = engle_granger_adf_stat(rw, max_lag=1)
    # A random walk should NOT clear the 5% critical value most of the time.
    assert stat > _EG_CV_5PCT, f"random walk shouldn't reject, got stat={stat}"


def test_adf_returns_inf_on_degenerate_input() -> None:
    assert engle_granger_adf_stat(np.zeros(3)) == float("inf")
    assert engle_granger_adf_stat(np.array([1.0, 2.0])) == float("inf")


def test_fit_pair_records_adf_outcome() -> None:
    n = 400
    rng = np.random.default_rng(2)
    rho = float(np.exp(-np.log(2.0) / 8.0))
    b_innov = rng.normal(0, 0.01, n)
    log_b = np.cumsum(b_innov)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + rng.normal(0, 0.01)
    log_a = 1.3 * log_b + 0.2 + eps
    idx = pd.bdate_range("2020-01-01", periods=n)
    fit = fit_pair(
        pd.Series(np.exp(log_a), index=idx, name="A"),
        pd.Series(np.exp(log_b), index=idx, name="B"),
    )
    assert fit is not None
    assert fit.adf_passes, f"expected ADF pass on cointegrated synth, got stat={fit.adf_stat}"


def test_kalman_recovers_static_beta_on_clean_series() -> None:
    rng = np.random.default_rng(3)
    n = 500
    log_b = np.cumsum(rng.normal(0, 0.01, n))
    eps = rng.normal(0, 0.01, n)
    log_a = 1.7 * log_b + 0.5 + eps
    fit = kalman_hedge(log_a, log_b, delta=1e-7, obs_var=1e-3)
    assert fit is not None
    # With tight process noise the Kalman β should track OLS closely.
    assert abs(fit.beta - 1.7) < 0.3


def test_kalman_tracks_time_varying_beta() -> None:
    """If the true β drifts mid-series, Kalman should follow."""
    rng = np.random.default_rng(4)
    n = 400
    log_b = np.cumsum(rng.normal(0, 0.01, n))
    log_a = np.zeros(n)
    # First half: β=1, second half: β=2.
    for t in range(n):
        true_beta = 1.0 if t < n // 2 else 2.0
        log_a[t] = true_beta * log_b[t] + rng.normal(0, 0.005)
    fit = kalman_hedge(log_a, log_b, delta=1e-4, obs_var=1e-3)
    assert fit is not None
    # Final β should be closer to 2 than to 1.
    assert fit.beta > 1.5, f"expected drift toward 2, got {fit.beta}"


def test_kalman_returns_none_on_degenerate_input() -> None:
    assert kalman_hedge(np.zeros(5), np.zeros(5)) is None
    assert kalman_hedge(np.zeros(50), np.zeros(50)) is None  # var_x = 0
