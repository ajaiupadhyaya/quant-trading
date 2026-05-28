"""Hand-rolled diagonal-covariance Gaussian HMM in numpy/scipy.

All recursions run in log-space for numerical stability. The *filter*
(forward-only) posterior is the only quantity safe for live decisions: it
conditions on obs[0..t] and never peeks ahead. Viterbi and forward-backward
use the full sample and are for offline analysis only.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp  # type: ignore[import-untyped]

from quant.regime.models import HMMParams

_LOG_2PI = float(np.log(2.0 * np.pi))


def log_emission(obs: np.ndarray, params: HMMParams) -> np.ndarray:
    """Per-state Gaussian log-density. obs (T, F) -> (T, K)."""
    x = np.asarray(obs, dtype=float)
    means = params.means  # (K, F)
    var = params.variances  # (K, F)
    # (T, 1, F) - (1, K, F) -> (T, K, F)
    diff = x[:, None, :] - means[None, :, :]
    log_det = np.log(var).sum(axis=1)  # (K,)
    quad = (diff**2 / var[None, :, :]).sum(axis=2)  # (T, K)
    n_features = x.shape[1]
    result: np.ndarray = -0.5 * (n_features * _LOG_2PI + log_det[None, :] + quad)
    return result


def forward_filter(obs: np.ndarray, params: HMMParams) -> np.ndarray:
    """Online filtered posteriors P(state_t | obs[0..t]). Returns (T, K)."""
    le = log_emission(obs, params)  # (T, K)
    log_trans = np.log(params.trans_mat)  # (K, K)
    log_start = np.log(params.start_prob)  # (K,)
    n_obs = le.shape[0]
    log_alpha = np.empty_like(le)
    log_alpha[0] = log_start + le[0]
    for t in range(1, n_obs):
        # log sum_i alpha[t-1, i] * trans[i, j]
        prev = log_alpha[t - 1][:, None] + log_trans  # (K, K)
        log_alpha[t] = np.asarray(logsumexp(prev, axis=0), dtype=float) + le[t]
    # Normalize each row to a posterior (subtract row logsumexp, exponentiate).
    log_post = log_alpha - np.asarray(logsumexp(log_alpha, axis=1, keepdims=True), dtype=float)
    posterior: np.ndarray = np.exp(log_post)
    return posterior
