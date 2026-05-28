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


def _forward_backward(
    le: np.ndarray, log_start: np.ndarray, log_trans: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (gamma (T,K), xi_sum (K,K), loglik). Full-sample (offline) smoothing."""
    n_obs, n_states = le.shape
    log_alpha = np.empty_like(le)
    log_alpha[0] = log_start + le[0]
    for t in range(1, n_obs):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + le[t]
    log_beta = np.zeros_like(le)
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + le[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    loglik = float(logsumexp(log_alpha[-1]))
    log_gamma = log_alpha + log_beta - loglik
    gamma = np.exp(log_gamma)
    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_obs - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + le[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - loglik
        )
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, loglik


def _fit_once(
    obs: np.ndarray,
    n_states: int,
    max_iter: int,
    tol: float,
    var_floor: float,
    rng: np.random.Generator,
) -> tuple[HMMParams, float]:
    n_obs, _n_features = obs.shape
    # Init means at random observations; variances at global variance; uniform trans.
    idx = rng.choice(n_obs, size=n_states, replace=False)
    means = obs[idx].copy()
    variances = np.tile(obs.var(axis=0) + var_floor, (n_states, 1))
    trans = np.full((n_states, n_states), 1.0 / n_states)
    start = np.full(n_states, 1.0 / n_states)

    prev_ll = -np.inf
    params = HMMParams(start, trans, means, variances)
    for _ in range(max_iter):
        le = log_emission(obs, params)
        gamma, xi_sum, loglik = _forward_backward(le, np.log(start), np.log(trans))
        if loglik - prev_ll < tol:
            return params, loglik
        prev_ll = loglik
        # M-step -> next params.
        start = gamma[0] / gamma[0].sum()
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        trans = np.divide(
            xi_sum, row_sums, out=np.full_like(xi_sum, 1.0 / n_states), where=row_sums > 0
        )
        weights = gamma.sum(axis=0)
        means = (gamma.T @ obs) / weights[:, None]
        diff2: np.ndarray = (obs[:, None, :] - means[None, :, :]) ** 2
        variances = np.maximum(np.einsum("tk,tkf->kf", gamma, diff2) / weights[:, None], var_floor)
        params = HMMParams(start, trans, means, variances)
    # max_iter exhausted: authoritative loglik of the params we return.
    le = log_emission(obs, params)
    _, _, loglik = _forward_backward(le, np.log(params.start_prob), np.log(params.trans_mat))
    return params, loglik


def fit_hmm(
    obs: np.ndarray,
    n_states: int = 3,
    n_restarts: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    var_floor: float = 1e-6,
) -> HMMParams:
    """Baum-Welch EM with multiple seeded restarts; keep the best log-likelihood."""
    x = np.asarray(obs, dtype=float)
    if x.ndim != 2 or x.shape[0] < n_states * 10:
        raise ValueError(f"fit_hmm needs a (T, F) array with T >= {n_states * 10}")
    best: HMMParams | None = None
    best_ll = -np.inf
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart)
        params, ll = _fit_once(x, n_states, max_iter, tol, var_floor, rng)
        if ll > best_ll:
            best_ll, best = ll, params
    if best is None:
        raise ValueError("fit_hmm requires n_restarts >= 1")
    return best


def viterbi(obs: np.ndarray, params: HMMParams) -> np.ndarray:
    """Most-likely state path (offline, uses full sample). Returns (T,) int array."""
    le = log_emission(obs, params)
    log_trans = np.log(params.trans_mat)
    n_obs, n_states = le.shape
    delta = np.empty_like(le)
    psi = np.zeros_like(le, dtype=int)
    delta[0] = np.log(params.start_prob) + le[0]
    for t in range(1, n_obs):
        scores = delta[t - 1][:, None] + log_trans  # (K, K)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(n_states)] + le[t]
    path = np.empty(n_obs, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def score(obs: np.ndarray, params: HMMParams) -> float:
    """Total log-likelihood of obs under params (forward recursion)."""
    le = log_emission(obs, params)
    log_trans = np.log(params.trans_mat)
    log_alpha = np.log(params.start_prob) + le[0]
    for t in range(1, le.shape[0]):
        log_alpha = logsumexp(log_alpha[:, None] + log_trans, axis=0) + le[t]
    return float(logsumexp(log_alpha))
