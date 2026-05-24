"""Monte Carlo bootstrap on a daily-returns series.

Two resamplers:
- ``iid_resample``: independent draws with replacement (destroys autocorr).
- ``stationary_block_resample``: Politis & Romano (1994) — geometrically-
  distributed block lengths, wraps around the end (circular). Preserves
  short-range serial correlation, which matters for Sharpe/drawdown CIs.

``bootstrap_ci`` returns 5/50/95 percentiles for total return, Sharpe, and
max drawdown across ``n_resamples`` stationary-block resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.metrics import max_drawdown, sharpe, total_return


@dataclass(frozen=True)
class BootstrapCI:
    total_return_p05: float
    total_return_median: float
    total_return_p95: float
    sharpe_p05: float
    sharpe_median: float
    sharpe_p95: float
    max_drawdown_p05: float
    max_drawdown_median: float
    max_drawdown_p95: float
    n_resamples: int


def iid_resample(returns: pd.Series, seed: int = 0) -> pd.Series:
    """Independent resample with replacement, preserving length and index."""
    n = len(returns)
    if n == 0:
        return returns.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n)
    return pd.Series(returns.to_numpy()[idx], index=returns.index)


def stationary_block_resample(
    returns: pd.Series, mean_block_len: int = 5, seed: int = 0
) -> pd.Series:
    """Politis-Romano stationary block bootstrap.

    Block lengths are drawn from Geometric(1/mean_block_len); blocks start at
    a uniformly-random offset and wrap around the series end (circular).
    """
    n = len(returns)
    if n == 0:
        return returns.copy()
    if mean_block_len < 1:
        raise ValueError(f"mean_block_len must be >= 1, got {mean_block_len}")

    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block_len
    arr = returns.to_numpy()

    out = np.empty(n, dtype=arr.dtype)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        # Geometric(p) here counts the number of Bernoulli trials until the
        # first success; numpy's geometric returns values >= 1, which is the
        # block length we want.
        block_len = int(rng.geometric(p))
        block_len = max(1, block_len)
        block_len = min(block_len, n - i)
        for k in range(block_len):
            out[i + k] = arr[(start + k) % n]
        i += block_len

    return pd.Series(out, index=returns.index)


def bootstrap_ci(
    returns: pd.Series,
    n_resamples: int = 1000,
    mean_block_len: int = 5,
    seed: int = 0,
) -> BootstrapCI:
    """Stationary-block bootstrap CIs for total return, Sharpe, and max DD."""
    if len(returns) == 0 or n_resamples <= 0:
        return BootstrapCI(
            total_return_p05=0.0,
            total_return_median=0.0,
            total_return_p95=0.0,
            sharpe_p05=0.0,
            sharpe_median=0.0,
            sharpe_p95=0.0,
            max_drawdown_p05=0.0,
            max_drawdown_median=0.0,
            max_drawdown_p95=0.0,
            n_resamples=0,
        )

    tr = np.empty(n_resamples, dtype=float)
    sr = np.empty(n_resamples, dtype=float)
    dd = np.empty(n_resamples, dtype=float)

    for k in range(n_resamples):
        sample = stationary_block_resample(returns, mean_block_len, seed=seed + k)
        tr[k] = total_return(sample)
        sr[k] = sharpe(sample)
        dd[k] = max_drawdown(sample)

    return BootstrapCI(
        total_return_p05=float(np.percentile(tr, 5)),
        total_return_median=float(np.percentile(tr, 50)),
        total_return_p95=float(np.percentile(tr, 95)),
        sharpe_p05=float(np.percentile(sr, 5)),
        sharpe_median=float(np.percentile(sr, 50)),
        sharpe_p95=float(np.percentile(sr, 95)),
        max_drawdown_p05=float(np.percentile(dd, 5)),
        max_drawdown_median=float(np.percentile(dd, 50)),
        max_drawdown_p95=float(np.percentile(dd, 95)),
        n_resamples=n_resamples,
    )
