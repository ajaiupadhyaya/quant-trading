"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado).

PSR is the probability that the true (per-period) Sharpe exceeds a benchmark.
DSR is PSR with the benchmark adjusted for the multiple-testing bias of
selecting the best among N trial strategies.

All Sharpe values in this module are per-period (un-annualized). Conversion
from a returns series uses the same ddof=1 convention as quant.backtest.metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore[import-untyped]

_STD_EPS = 1e-12
_EULER_MASCHERONI = 0.5772156649015329


def _sr_period_from_returns(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= _STD_EPS:
        return 0.0
    return float(returns.mean()) / std


def _skew_kurt(returns: pd.Series) -> tuple[float, float]:
    """Return (skew, kurtosis_non_excess). Falls back to (0, 3) if undefined."""
    n = len(returns)
    if n < 4:
        return 0.0, 3.0
    arr = returns.to_numpy(dtype=float)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma <= _STD_EPS:
        return 0.0, 3.0
    z = (arr - mu) / sigma
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # non-excess
    return skew, kurt


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float) -> float:
    """Probabilistic Sharpe Ratio against a per-period benchmark.

    Returns Pr(true SR > sr_benchmark) given the observed sample. 0.0 on empty
    input or zero volatility.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    sr = _sr_period_from_returns(returns)
    if sr == 0.0 and float(returns.std(ddof=1)) <= _STD_EPS:
        return 0.0
    skew, kurt = _skew_kurt(returns)
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0.0:
        return 0.0
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe(returns: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio.

    ``trial_sharpes`` is an array of per-period Sharpe ratios from every
    backtest trial run during model selection (e.g., the cartesian-product
    grid in walk-forward). Returns 0.0 on empty inputs.
    """
    n = len(returns)
    if n < 2 or len(trial_sharpes) == 0:
        return 0.0

    n_trials = len(trial_sharpes)
    sr_trial_var = float(np.var(trial_sharpes, ddof=1)) if n_trials > 1 else 0.0
    sr_trial_std = float(np.sqrt(max(sr_trial_var, 0.0)))

    # Expected maximum of N i.i.d. standard normals (Lopez de Prado 2018):
    # E[max] ~= (1 - gamma) * invCDF(1 - 1/N) + gamma * invCDF(1 - 1/(N e))
    # where gamma is the Euler-Mascheroni constant.
    if n_trials <= 1:
        expected_max_z = 0.0
    else:
        e = np.e
        expected_max_z = (1.0 - _EULER_MASCHERONI) * float(norm.ppf(1.0 - 1.0 / n_trials)) + (
            _EULER_MASCHERONI * float(norm.ppf(1.0 - 1.0 / (n_trials * e)))
        )

    sr_zero = sr_trial_std * expected_max_z
    return probabilistic_sharpe(returns, sr_benchmark=sr_zero)
