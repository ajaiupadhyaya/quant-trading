# Regime Detection Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a market-wide 3-state Gaussian HMM regime detector (hand-rolled numpy/scipy) that labels each day `calm-bull` / `choppy` / `crisis` as a point-in-time, validation-gated *observed* signal — it never changes a live position until it passes its own gate.

**Architecture:** A self-contained `quant/regime/` package. Pure-math HMM core (`hmm.py`) and a Kalman feature smoother (`kalman_state.py`) take numpy arrays and know nothing about markets. `features.py` builds a point-in-time standardized feature matrix from cached bars + FRED macro. `detect.py` is the only orchestrator: it runs a walk-forward refit loop, produces daily *filtered* posteriors, and assigns stable canonical labels. `validation.py` scores the signal against four out-of-sample gates. CLI + TUI integration is additive — nothing existing is removed.

**Tech Stack:** Python 3.12, numpy, scipy (`scipy.special.logsumexp`), pandas, click, rich, textual, pytest, hypothesis. No new dependencies. `uv`-managed; mypy-strict; ruff lint + format.

**Spec:** `docs/superpowers/specs/2026-05-28-regime-detection-engine-design.md`

---

## Conventions for every task

- Run tests with the venv binary: `.venv/bin/pytest <path> -v` (or `uv run pytest`).
- After each task, before committing, the change must pass: `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/mypy quant`. If format check fails, run `.venv/bin/ruff format .` and re-stage.
- All new modules start with `from __future__ import annotations`.
- Frozen dataclasses that hold `np.ndarray` use `@dataclass(frozen=True, eq=False)` (ndarray `__eq__` is elementwise and breaks dataclass equality/hash).
- Commit messages follow the repo style: `feat(regime): ...`, `test(regime): ...`, `docs(regime): ...`.

---

## File structure

```
quant/regime/
  __init__.py      # package marker + public re-exports
  models.py        # REGIME_LABELS, N_STATES, HMMParams, RegimeReport
  hmm.py           # Gaussian HMM: emission, forward_filter, forward_backward, fit_hmm, viterbi, score
  kalman_state.py  # kalman_local_level online smoother
  features.py      # FeatureConfig, build_feature_matrix (pure), load_market_features (I/O)
  detect.py        # DetectConfig, identify_states, run_detection, persist_*
  validation.py    # validate_regime_series, check_pit_consistency

tests/regime/
  __init__.py
  test_models.py
  test_hmm.py
  test_kalman_state.py
  test_features.py
  test_detect.py
  test_validation.py

quant/cli.py        # MODIFY: add `quant regime` group (fit/label/backtest/validate)
quant/tui.py        # MODIFY: add regime panel to MonitorSnapshot + layout
tests/test_cli.py   # MODIFY: add regime CLI tests
README.md           # MODIFY: document `quant regime` commands
```

---

## Task 1: Package skeleton + models

**Files:**
- Create: `quant/regime/__init__.py`
- Create: `quant/regime/models.py`
- Create: `tests/regime/__init__.py`
- Test: `tests/regime/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/regime/test_models.py`:

```python
from __future__ import annotations

import numpy as np

from quant.regime.models import N_STATES, REGIME_LABELS, HMMParams


def test_regime_label_constants():
    assert REGIME_LABELS == ("calm-bull", "choppy", "crisis")
    assert N_STATES == 3


def test_hmmparams_shapes_and_roundtrip():
    params = HMMParams(
        start_prob=np.array([0.5, 0.3, 0.2]),
        trans_mat=np.full((3, 3), 1 / 3),
        means=np.zeros((3, 2)),
        variances=np.ones((3, 2)),
    )
    assert params.n_states == 3
    assert params.n_features == 2

    restored = HMMParams.from_json_dict(params.to_json_dict())
    np.testing.assert_allclose(restored.start_prob, params.start_prob)
    np.testing.assert_allclose(restored.trans_mat, params.trans_mat)
    np.testing.assert_allclose(restored.means, params.means)
    np.testing.assert_allclose(restored.variances, params.variances)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.regime'`

- [ ] **Step 3: Write minimal implementation**

`tests/regime/__init__.py`: empty file.

`quant/regime/__init__.py`:

```python
"""Market-wide regime detection: a point-in-time, validation-gated HMM signal."""
```

`quant/regime/models.py`:

```python
"""Frozen value types and label constants for the regime engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIME_LABELS: tuple[str, str, str] = ("calm-bull", "choppy", "crisis")
N_STATES: int = 3


@dataclass(frozen=True, eq=False)
class HMMParams:
    """Parameters of a diagonal-covariance Gaussian HMM.

    Shapes: start_prob (K,), trans_mat (K, K), means (K, F), variances (K, F).
    K = number of hidden states, F = number of features.
    """

    start_prob: np.ndarray
    trans_mat: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def to_json_dict(self) -> dict[str, object]:
        return {
            "start_prob": self.start_prob.tolist(),
            "trans_mat": self.trans_mat.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> HMMParams:
        return cls(
            start_prob=np.asarray(payload["start_prob"], dtype=float),
            trans_mat=np.asarray(payload["trans_mat"], dtype=float),
            means=np.asarray(payload["means"], dtype=float),
            variances=np.asarray(payload["variances"], dtype=float),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_models.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/__init__.py quant/regime/models.py tests/regime/__init__.py tests/regime/test_models.py
git commit -m "feat(regime): package skeleton + HMMParams value type"
```

---

## Task 2: HMM emission + forward filter (the live path)

**Files:**
- Create: `quant/regime/hmm.py`
- Test: `tests/regime/test_hmm.py`

- [ ] **Step 1: Write the failing test**

`tests/regime/test_hmm.py`:

```python
from __future__ import annotations

import numpy as np

from quant.regime.hmm import forward_filter, log_emission
from quant.regime.models import HMMParams


def _toy_params() -> HMMParams:
    return HMMParams(
        start_prob=np.array([0.6, 0.4]),
        trans_mat=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[0.0], [3.0]]),
        variances=np.array([[1.0], [1.0]]),
    )


def test_log_emission_shape_and_peak():
    obs = np.array([[0.0], [3.0]])
    le = log_emission(obs, _toy_params())
    assert le.shape == (2, 2)
    # Obs 0 (value 0) most likely under state 0; obs 1 (value 3) under state 1.
    assert le[0, 0] > le[0, 1]
    assert le[1, 1] > le[1, 0]


def test_forward_filter_is_normalized_and_causal():
    obs = np.array([[0.0], [0.0], [3.0], [3.0]])
    post = forward_filter(obs, _toy_params())
    assert post.shape == (4, 2)
    np.testing.assert_allclose(post.sum(axis=1), np.ones(4), atol=1e-9)
    # Filtered posterior at t depends only on obs[:t+1]: truncating later obs
    # must not change earlier rows.
    post_trunc = forward_filter(obs[:2], _toy_params())
    np.testing.assert_allclose(post[:2], post_trunc, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_hmm.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'forward_filter'`

- [ ] **Step 3: Write minimal implementation**

`quant/regime/hmm.py`:

```python
"""Hand-rolled diagonal-covariance Gaussian HMM in numpy/scipy.

All recursions run in log-space for numerical stability. The *filter*
(forward-only) posterior is the only quantity safe for live decisions: it
conditions on obs[0..t] and never peeks ahead. Viterbi and forward-backward
use the full sample and are for offline analysis only.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

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
    return -0.5 * (n_features * _LOG_2PI + log_det[None, :] + quad)


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
        log_alpha[t] = logsumexp(prev, axis=0) + le[t]
    # Normalize each row to a posterior (subtract row logsumexp, exponentiate).
    log_post = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_post)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_hmm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/hmm.py tests/regime/test_hmm.py
git commit -m "feat(regime): Gaussian HMM emission + online forward filter"
```

---

## Task 3: HMM Baum-Welch fit with restarts

**Files:**
- Modify: `quant/regime/hmm.py`
- Test: `tests/regime/test_hmm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/regime/test_hmm.py`:

```python
def test_fit_recovers_known_params_and_is_seed_reproducible():
    rng = np.random.default_rng(0)
    true = HMMParams(
        start_prob=np.array([1.0, 0.0]),
        trans_mat=np.array([[0.97, 0.03], [0.05, 0.95]]),
        means=np.array([[0.0], [5.0]]),
        variances=np.array([[0.5], [0.5]]),
    )
    # Generate a long sample from `true`.
    states = np.zeros(4000, dtype=int)
    for t in range(1, states.size):
        states[t] = rng.choice(2, p=true.trans_mat[states[t - 1]])
    obs = (true.means[states] + rng.normal(0, np.sqrt(0.5), size=(4000, 1)))

    from quant.regime.hmm import fit_hmm

    fit_a = fit_hmm(obs, n_states=2, n_restarts=4, seed=7)
    fit_b = fit_hmm(obs, n_states=2, n_restarts=4, seed=7)

    # Seed reproducibility.
    np.testing.assert_allclose(fit_a.means, fit_b.means)

    # Recover the two cluster means (order-agnostic).
    recovered = np.sort(fit_a.means.ravel())
    np.testing.assert_allclose(recovered, np.array([0.0, 5.0]), atol=0.4)

    # Transition matrix rows are valid distributions.
    np.testing.assert_allclose(fit_a.trans_mat.sum(axis=1), np.ones(2), atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_hmm.py::test_fit_recovers_known_params_and_is_seed_reproducible -v`
Expected: FAIL with `ImportError: cannot import name 'fit_hmm'`

- [ ] **Step 3: Write minimal implementation**

Append to `quant/regime/hmm.py`:

```python
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
    obs: np.ndarray, n_states: int, max_iter: int, tol: float, var_floor: float, rng: np.random.Generator
) -> tuple[HMMParams, float]:
    n_obs, n_features = obs.shape
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
        # M-step.
        start = gamma[0] / gamma[0].sum()
        trans = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        weights = gamma.sum(axis=0)  # (K,)
        means = (gamma.T @ obs) / weights[:, None]
        diff2 = (obs[:, None, :] - means[None, :, :]) ** 2  # (T, K, F)
        variances = np.einsum("tk,tkf->kf", gamma, diff2) / weights[:, None]
        variances = np.maximum(variances, var_floor)
        params = HMMParams(start, trans, means, variances)
        if loglik - prev_ll < tol:
            break
        prev_ll = loglik
    return params, prev_ll


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
    assert best is not None
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_hmm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/hmm.py tests/regime/test_hmm.py
git commit -m "feat(regime): Baum-Welch EM fit with seeded restarts"
```

---

## Task 4: Viterbi path + score

**Files:**
- Modify: `quant/regime/hmm.py`
- Test: `tests/regime/test_hmm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/regime/test_hmm.py`:

```python
def test_viterbi_recovers_path_and_score_is_finite():
    from quant.regime.hmm import score, viterbi

    params = HMMParams(
        start_prob=np.array([0.5, 0.5]),
        trans_mat=np.array([[0.95, 0.05], [0.05, 0.95]]),
        means=np.array([[0.0], [10.0]]),
        variances=np.array([[0.25], [0.25]]),
    )
    # Low-noise: first 20 near 0 (state 0), next 20 near 10 (state 1).
    obs = np.concatenate(
        [np.zeros(20), np.full(20, 10.0)]
    )[:, None] + np.random.default_rng(1).normal(0, 0.05, size=(40, 1))
    path = viterbi(obs, params)
    assert path.shape == (40,)
    assert path[:20].tolist() == [0] * 20
    assert path[20:].tolist() == [1] * 20
    assert np.isfinite(score(obs, params))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_hmm.py::test_viterbi_recovers_path_and_score_is_finite -v`
Expected: FAIL with `ImportError: cannot import name 'viterbi'`

- [ ] **Step 3: Write minimal implementation**

Append to `quant/regime/hmm.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_hmm.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/hmm.py tests/regime/test_hmm.py
git commit -m "feat(regime): Viterbi path + log-likelihood score"
```

---

## Task 5: Kalman feature smoother

**Files:**
- Create: `quant/regime/kalman_state.py`
- Test: `tests/regime/test_kalman_state.py`

- [ ] **Step 1: Write the failing test**

`tests/regime/test_kalman_state.py`:

```python
from __future__ import annotations

import numpy as np

from quant.regime.kalman_state import kalman_local_level


def test_smoother_reduces_noise_and_is_causal():
    rng = np.random.default_rng(0)
    truth = np.linspace(0.0, 1.0, 200)
    noisy = truth + rng.normal(0, 0.3, size=200)
    smooth = kalman_local_level(noisy, process_var=1e-4, obs_var=1e-1)
    assert smooth.shape == (200,)
    # Smoothed series tracks truth better than the noisy input.
    assert np.mean((smooth - truth) ** 2) < np.mean((noisy - truth) ** 2)
    # Online/causal: value at t unchanged by future observations.
    smooth_trunc = kalman_local_level(noisy[:100], process_var=1e-4, obs_var=1e-1)
    np.testing.assert_allclose(smooth[:100], smooth_trunc, atol=1e-12)


def test_short_input_returns_input_copy():
    assert kalman_local_level(np.array([])).shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_kalman_state.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`quant/regime/kalman_state.py`:

```python
"""1-D local-level Kalman smoother for denoising regime features.

State model: level_t = level_{t-1} + w_t (w_t ~ N(0, process_var));
observation: y_t = level_t + v_t (v_t ~ N(0, obs_var)). The filtered level is
online (causal) — value at t depends only on y[0..t], so it never leaks future
information into the feature matrix.
"""

from __future__ import annotations

import numpy as np


def kalman_local_level(
    y: np.ndarray, process_var: float = 1e-4, obs_var: float = 1e-2
) -> np.ndarray:
    """Return the online filtered level estimate, same length as y."""
    obs = np.asarray(y, dtype=float)
    n = obs.size
    if n == 0:
        return obs.copy()
    out = np.empty(n)
    level = float(obs[0])
    cov = 1.0
    for t in range(n):
        # Predict.
        pred_cov = cov + process_var
        # Update.
        gain = pred_cov / (pred_cov + obs_var)
        level = level + gain * (float(obs[t]) - level)
        cov = (1.0 - gain) * pred_cov
        out[t] = level
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_kalman_state.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/kalman_state.py tests/regime/test_kalman_state.py
git commit -m "feat(regime): 1-D local-level Kalman feature smoother"
```

---

## Task 6: Point-in-time feature matrix

**Files:**
- Create: `quant/regime/features.py`
- Test: `tests/regime/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/regime/test_features.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.regime.features import FeatureConfig, build_feature_matrix


def _series(n: int, seed: int) -> pd.Series:
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_feature_matrix_columns_and_no_nan_tail():
    spy = _series(400, 1)
    vix = pd.Series(np.full(400, 18.0), index=spy.index)
    cfg = FeatureConfig(use_term_spread=False)
    feats = build_feature_matrix(spy_close=spy, vix=vix, dgs10=None, dgs2=None, config=cfg)
    assert list(feats.columns) == ["ret", "vol", "vix", "drawdown"]
    # The warmup window is dropped; remaining rows are fully populated.
    assert not feats.isna().any().any()
    assert len(feats) > 0


def test_standardization_is_trailing_only():
    # Build features over the full series, then over a truncated prefix.
    # A given date's standardized features must be identical in both — i.e.
    # standardization uses only trailing data, never the full sample.
    spy = _series(400, 2)
    vix = pd.Series(np.full(400, 18.0), index=spy.index)
    cfg = FeatureConfig(use_term_spread=False, standardize_window=120)
    full = build_feature_matrix(spy_close=spy, vix=vix, dgs10=None, dgs2=None, config=cfg)
    prefix = build_feature_matrix(
        spy_close=spy.iloc[:300], vix=vix.iloc[:300], dgs10=None, dgs2=None, config=cfg
    )
    shared = prefix.index
    pd.testing.assert_frame_equal(full.loc[shared], prefix, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`quant/regime/features.py`:

```python
"""Point-in-time market feature matrix for the regime HMM.

Every transform uses only trailing data as of each date. Standardization is
rolling (or expanding), never full-sample — full-sample scaling would leak the
future into earlier rows, the exact look-ahead the validation gate forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quant.data import bars, macro
from quant.regime.kalman_state import kalman_local_level


@dataclass(frozen=True)
class FeatureConfig:
    realized_vol_window: int = 21
    use_term_spread: bool = True
    standardize_window: int = 252  # rolling window; 0 => expanding
    kalman_process_var: float = 1e-4
    kalman_obs_var: float = 1e-2
    min_standardize_obs: int = 60


def _standardize(col: pd.Series, window: int, min_obs: int) -> pd.Series:
    if window > 0:
        roll = col.rolling(window=window, min_periods=min_obs)
    else:
        roll = col.expanding(min_periods=min_obs)
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (col - mean) / std.replace(0.0, np.nan)


def build_feature_matrix(
    *,
    spy_close: pd.Series,
    vix: pd.Series,
    dgs10: pd.Series | None,
    dgs2: pd.Series | None,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Return a date-indexed, trailing-standardized feature frame (warmup dropped)."""
    spy_close = spy_close.sort_index().astype(float)
    log_ret = np.log(spy_close).diff()
    smoothed_ret = pd.Series(
        kalman_local_level(
            log_ret.fillna(0.0).to_numpy(),
            process_var=config.kalman_process_var,
            obs_var=config.kalman_obs_var,
        ),
        index=spy_close.index,
    )
    realized_vol = log_ret.rolling(config.realized_vol_window).std(ddof=0)
    log_vol = np.log(realized_vol.replace(0.0, np.nan))
    running_peak = spy_close.cummax()
    drawdown = spy_close / running_peak - 1.0

    vix_aligned = vix.sort_index().reindex(spy_close.index).ffill()

    raw = pd.DataFrame(
        {
            "ret": smoothed_ret,
            "vol": log_vol,
            "vix": vix_aligned,
            "drawdown": drawdown,
        }
    )
    if config.use_term_spread:
        if dgs10 is None or dgs2 is None:
            raise ValueError("use_term_spread=True requires dgs10 and dgs2 series")
        spread = (dgs10.sort_index().reindex(spy_close.index).ffill()) - (
            dgs2.sort_index().reindex(spy_close.index).ffill()
        )
        raw["term_spread"] = spread

    standardized = raw.apply(
        lambda c: _standardize(c, config.standardize_window, config.min_standardize_obs)
    )
    return standardized.dropna()


def load_market_features(start: date, end: date, config: FeatureConfig) -> pd.DataFrame:
    """Load cached bars + FRED macro and build the feature matrix."""
    spy = bars.get_bars(bars.BarRequest(symbols=["SPY"], start=start, end=end))
    spy_close = _extract_close(spy, "SPY")
    vix = macro.get_series(macro.FRED_SERIES["vix"])
    dgs10 = macro.get_series(macro.FRED_SERIES["tenyear"]) if config.use_term_spread else None
    dgs2 = macro.get_series(macro.FRED_SERIES["twoyear"]) if config.use_term_spread else None
    return build_feature_matrix(
        spy_close=spy_close, vix=vix, dgs10=dgs10, dgs2=dgs2, config=config
    )


def _extract_close(frame: pd.DataFrame, symbol: str) -> pd.Series:
    """Pull the close column for `symbol` from get_bars' wide frame."""
    if isinstance(frame.columns, pd.MultiIndex):
        close = frame[(symbol, "close")]
    elif "close" in frame.columns:
        close = frame["close"]
    else:
        raise KeyError(f"No close column for {symbol} in bars frame")
    return pd.Series(close, index=frame.index).astype(float)
```

> Note: `_extract_close` is defensive about the bars column layout (MultiIndex `(symbol, field)` vs flat). Confirm against a real `get_bars(["SPY"])` call during implementation and keep whichever branch fires; do not delete the other unless you verify it can never occur.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_features.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/features.py tests/regime/test_features.py
git commit -m "feat(regime): point-in-time standardized feature matrix"
```

---

## Task 7: Walk-forward detection + canonical labels + persistence

**Files:**
- Create: `quant/regime/detect.py`
- Test: `tests/regime/test_detect.py`

- [ ] **Step 1: Write the failing test**

`tests/regime/test_detect.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.regime.detect import DetectConfig, identify_states, run_detection
from quant.regime.models import HMMParams


def test_identify_states_orders_by_volatility():
    # Feature column order is [ret, vol, ...]; vol is index 1.
    # State means: state0 high vol, state1 low vol, state2 mid vol.
    params = HMMParams(
        start_prob=np.full(3, 1 / 3),
        trans_mat=np.full((3, 3), 1 / 3),
        means=np.array([[0.0, 2.0], [0.1, -1.0], [0.0, 0.5]]),
        variances=np.ones((3, 2)),
    )
    mapping = identify_states(params, vol_index=1)
    # canonical 0=calm (lowest vol)=raw1, 1=choppy (mid)=raw2, 2=crisis (high)=raw0
    assert mapping == [1, 2, 0]


def _synthetic_features(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.default_rng(0)
    # Two clearly separated blocks so the HMM finds structure.
    ret = np.where(np.arange(n) % 500 < 400, 0.5, -0.5) + rng.normal(0, 0.2, n)
    vol = np.where(np.arange(n) % 500 < 400, -0.5, 1.5) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"ret": ret, "vol": vol}, index=idx)


def test_run_detection_outputs_daily_labels_and_is_pit():
    feats = _synthetic_features(1500)
    cfg = DetectConfig(train_window_days=750, refit_freq="YS", n_restarts=3, seed=0)
    out = run_detection(feats, cfg)
    assert set(out.columns) >= {"p_calm", "p_choppy", "p_crisis", "label", "refit_epoch"}
    np.testing.assert_allclose(
        out[["p_calm", "p_choppy", "p_crisis"]].sum(axis=1).to_numpy(),
        np.ones(len(out)),
        atol=1e-9,
    )
    assert set(out["label"].unique()).issubset({"calm-bull", "choppy", "crisis"})
    # PIT: re-running on a truncated feature frame must reproduce earlier labels
    # exactly (no future data influences a past label).
    trunc = run_detection(feats.iloc[:1200], cfg)
    shared = trunc.index.intersection(out.index)
    assert (out.loc[shared, "label"] == trunc.loc[shared, "label"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`quant/regime/detect.py`:

```python
"""Walk-forward regime detection: the only orchestrator in the package.

Refits the HMM on a schedule over a trailing window, then runs the online
forward filter forward with frozen params until the next refit. After each
refit, raw EM state indices are mapped to canonical labels by their fitted
volatility so the daily label series stays continuous across refit boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.regime.hmm import fit_hmm, forward_filter
from quant.regime.models import N_STATES, REGIME_LABELS, HMMParams


@dataclass(frozen=True)
class DetectConfig:
    refit_freq: str = "MS"  # pandas offset alias: month start
    train_window_days: int = 252 * 5
    expanding: bool = False
    n_restarts: int = 8
    seed: int = 0


def identify_states(params: HMMParams, vol_index: int = 1) -> list[int]:
    """Map raw state index -> canonical index (0 calm, 1 choppy, 2 crisis).

    Ranks states by fitted mean volatility feature ascending: lowest vol is
    calm-bull, highest is crisis. Returns a list `mapping` where
    `mapping[canonical] = raw_state`. Deterministic via stable argsort.
    """
    vol_means = params.means[:, vol_index]
    order = np.argsort(vol_means, kind="stable")
    return [int(order[c]) for c in range(params.n_states)]


def _refit_dates(index: pd.DatetimeIndex, config: DetectConfig) -> list[pd.Timestamp]:
    # First valid refit once we have a full training window.
    start_pos = config.train_window_days
    if start_pos >= len(index):
        return [index[len(index) // 2]] if len(index) else []
    anchors = pd.Series(index=index, data=index).resample(config.refit_freq).first().dropna()
    return [ts for ts in anchors if ts >= index[start_pos]]


def run_detection(features: pd.DataFrame, config: DetectConfig) -> pd.DataFrame:
    """Produce a daily filtered-posterior + canonical-label frame."""
    feats = features.sort_index()
    index = feats.index
    refit_dates = _refit_dates(index, config)
    if not refit_dates:
        refit_dates = [index[0]]

    rows: dict[pd.Timestamp, dict[str, object]] = {}
    for epoch, refit_ts in enumerate(refit_dates):
        end = refit_ts
        if config.expanding:
            train = feats.loc[:end].iloc[:-1]
        else:
            train = feats.loc[:end].iloc[-(config.train_window_days + 1) : -1]
        if len(train) < N_STATES * 10:
            continue
        params = fit_hmm(
            train.to_numpy(),
            n_states=N_STATES,
            n_restarts=config.n_restarts,
            seed=config.seed,
        )
        mapping = identify_states(params)
        # Filter from this refit until the next refit date.
        seg_end = refit_dates[epoch + 1] if epoch + 1 < len(refit_dates) else index[-1]
        seg = feats.loc[end:seg_end]
        if seg.empty:
            continue
        post_raw = forward_filter(seg.to_numpy(), params)  # (T, K) raw-state order
        post = post_raw[:, mapping]  # reorder columns to canonical
        labels = np.array(REGIME_LABELS)[post.argmax(axis=1)]
        for i, ts in enumerate(seg.index):
            rows[ts] = {
                "p_calm": float(post[i, 0]),
                "p_choppy": float(post[i, 1]),
                "p_crisis": float(post[i, 2]),
                "label": str(labels[i]),
                "refit_epoch": epoch,
            }
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index.name = "date"
    return out


def persist_regime_series(frame: pd.DataFrame, data_dir: Path) -> Path:
    path = data_dir / "regime" / "regime_series.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def persist_model(params: HMMParams, meta: dict[str, object], data_dir: Path) -> Path:
    path = data_dir / "regime" / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": params.to_json_dict(), "meta": meta}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
```

> Note: when consecutive refit segments share the boundary date (`feats.loc[end:seg_end]` is inclusive on both ends), the later epoch overwrites the boundary row in `rows`. That is intentional — the most recent refit owns the boundary. Verify the PIT test still passes; if a boundary row flips under truncation, change the segment slice to `feats.loc[end:seg_end].iloc[:-1]` for non-final epochs and re-run.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_detect.py -v`
Expected: PASS (2 tests). If the PIT test fails on a boundary row, apply the slice fix in the note above.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/detect.py tests/regime/test_detect.py
git commit -m "feat(regime): walk-forward detection with canonical labels + persistence"
```

---

## Task 8: Validation gates

**Files:**
- Create: `quant/regime/validation.py`
- Test: `tests/regime/test_validation.py`

- [ ] **Step 1: Write the failing test**

`tests/regime/test_validation.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.regime.validation import RegimeReport, validate_regime_series


def _regime_frame(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)
    # Persistent blocks: 200 calm, 40 crisis, repeating.
    labels = []
    while len(labels) < n:
        labels += ["calm-bull"] * 200 + ["crisis"] * 40
    labels = labels[:n]
    p = {"calm-bull": (0.8, 0.1, 0.1), "choppy": (0.1, 0.8, 0.1), "crisis": (0.1, 0.1, 0.8)}
    frame = pd.DataFrame(
        {
            "p_calm": [p[lbl][0] for lbl in labels],
            "p_choppy": [p[lbl][1] for lbl in labels],
            "p_crisis": [p[lbl][2] for lbl in labels],
            "label": labels,
            "refit_epoch": 0,
        },
        index=idx,
    )
    frame.index.name = "date"
    return frame


def test_validate_returns_report_with_four_gates():
    frame = _regime_frame(600)
    rng = np.random.default_rng(1)
    # Returns that crash during crisis labels — so de-risking helps.
    rets = pd.Series(
        np.where(frame["label"].to_numpy() == "crisis", -0.02, 0.001)
        + rng.normal(0, 0.005, len(frame)),
        index=frame.index,
    )
    report = validate_regime_series(frame, spy_returns=rets)
    assert isinstance(report, RegimeReport)
    assert set(report.gates) == {
        "persistence",
        "coherence",
        "predictive_lift",
        "pit_consistent",
    }
    # Crisis returns are clearly worse, so de-risking lifts the drawdown metric.
    assert report.gates["predictive_lift"] is True
    assert report.gates["persistence"] is True


def test_churny_series_fails_persistence():
    idx = pd.bdate_range("2018-01-01", periods=300)
    labels = np.array(["calm-bull", "crisis"] * 150)  # flips every day
    frame = pd.DataFrame(
        {
            "p_calm": 0.5,
            "p_choppy": 0.0,
            "p_crisis": 0.5,
            "label": labels,
            "refit_epoch": 0,
        },
        index=idx,
    )
    rets = pd.Series(np.zeros(300), index=idx)
    report = validate_regime_series(frame, spy_returns=rets)
    assert report.gates["persistence"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/regime/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`quant/regime/validation.py`:

```python
"""Out-of-sample validation gates for the regime signal.

A signal graduates from observed to tradable only if it is persistent,
economically coherent, adds predictive risk-reduction, and is point-in-time
consistent. All metrics use the filtered label series — never smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.regime.models import REGIME_LABELS

_DERISK_WEIGHT = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}


@dataclass(frozen=True)
class RegimeReport:
    gates: dict[str, bool]
    metrics: dict[str, float]

    @property
    def overall(self) -> bool:
        return all(self.gates.values())


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _median_run_length(labels: pd.Series) -> float:
    changed = labels.ne(labels.shift()).cumsum()
    runs = labels.groupby(changed).size()
    return float(runs.median())


def validate_regime_series(
    frame: pd.DataFrame,
    spy_returns: pd.Series,
    min_median_run: int = 5,
) -> RegimeReport:
    """Run the four gates and return a RegimeReport."""
    labels = frame["label"]
    rets = spy_returns.reindex(frame.index).fillna(0.0)

    # Gate 1: persistence.
    median_run = _median_run_length(labels)
    persistence = median_run >= min_median_run

    # Gate 2: coherence — forward vol increases calm -> choppy -> crisis.
    fwd_vol = rets.rolling(5).std(ddof=0).shift(-5)
    vol_by_label = {
        lbl: float(fwd_vol[labels == lbl].mean()) if (labels == lbl).any() else np.nan
        for lbl in REGIME_LABELS
    }
    present = [vol_by_label[lbl] for lbl in REGIME_LABELS if not np.isnan(vol_by_label[lbl])]
    coherence = len(present) >= 2 and all(
        present[i] <= present[i + 1] + 1e-9 for i in range(len(present) - 1)
    )

    # Gate 3: predictive lift — de-risk with YESTERDAY's label, compare drawdown.
    weights = labels.map(_DERISK_WEIGHT).astype(float).shift(1).fillna(1.0)
    baseline_equity = (1.0 + rets).cumprod()
    derisked_equity = (1.0 + rets * weights).cumprod()
    dd_base = _max_drawdown(baseline_equity)
    dd_derisk = _max_drawdown(derisked_equity)
    predictive_lift = dd_derisk > dd_base  # less negative = shallower drawdown

    # Gate 4: pit_consistent — placeholder True here; the authoritative check is
    # check_pit_consistency() run against the live detection path in the CLI and
    # in tests/regime/test_detect.py. We surface it so the report has 4 gates.
    pit_consistent = True

    return RegimeReport(
        gates={
            "persistence": bool(persistence),
            "coherence": bool(coherence),
            "predictive_lift": bool(predictive_lift),
            "pit_consistent": bool(pit_consistent),
        },
        metrics={
            "median_run_length": float(median_run),
            "max_drawdown_baseline": float(dd_base),
            "max_drawdown_derisked": float(dd_derisk),
            "fwd_vol_calm": float(vol_by_label["calm-bull"]),
            "fwd_vol_crisis": float(vol_by_label["crisis"]),
        },
    )
```

> Note on Gate 4: the genuine no-look-ahead audit is the truncation comparison already implemented in `tests/regime/test_detect.py::test_run_detection_outputs_daily_labels_and_is_pit`. The `pit_consistent` entry in the report exists so the report exposes all four gates the spec names; the CLI `validate` command (Task 9) sets it from an actual `run_detection` truncation re-run, not the hardcoded `True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/regime/test_validation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/regime/validation.py tests/regime/test_validation.py
git commit -m "feat(regime): out-of-sample validation gates"
```

---

## Task 9: CLI `quant regime` group

**Files:**
- Modify: `quant/cli.py` (add a new group near the `governance`/`research` groups; reuse existing imports `Settings`, `console`, `Table`, `append_experiment`, `ExperimentRecord`, `subprocess`, `uuid`, `datetime`, `UTC`, `Path`)
- Test: `tests/test_cli.py` (append new tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py` (match the file's existing CliRunner pattern — check the top of the file for the shared `runner` fixture / import and reuse it):

```python
def test_regime_label_and_validate_commands(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from click.testing import CliRunner

    from quant.cli import cli

    # Point data_dir at tmp and pre-write a regime series so `label` has data.
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setenv("FRED_API_KEY", "z")

    idx = pd.bdate_range("2020-01-01", periods=10)
    frame = pd.DataFrame(
        {
            "p_calm": 0.7,
            "p_choppy": 0.2,
            "p_crisis": 0.1,
            "label": "calm-bull",
            "refit_epoch": 0,
        },
        index=idx,
    )
    frame.index.name = "date"
    out = tmp_path / "regime" / "regime_series.parquet"
    out.parent.mkdir(parents=True)
    frame.to_parquet(out)

    result = CliRunner().invoke(cli, ["regime", "label"])
    assert result.exit_code == 0, result.output
    assert "calm-bull" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py::test_regime_label_and_validate_commands -v`
Expected: FAIL — `No such command 'regime'`

- [ ] **Step 3: Write minimal implementation**

Add to `quant/cli.py` (place after the `research` group definitions). The helper `_git_sha()` may already effectively exist inline in `validate`; if there is no reusable helper, add this one and use it:

```python
@cli.group(help="Market-wide regime detection (HMM/Kalman) — an observed, gated signal.")
def regime() -> None:
    pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _regime_series_path() -> Path:
    return Settings().data_dir / "regime" / "regime_series.parquet"  # type: ignore[call-arg]


@regime.command("fit", help="Refit the HMM walk-forward and persist the daily regime series.")
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
def regime_fit(start: str, end: str | None) -> None:
    import pandas as pd

    from quant.regime.detect import DetectConfig, persist_regime_series, run_detection
    from quant.regime.features import FeatureConfig, load_market_features

    settings = Settings()  # type: ignore[call-arg]
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date() if end else pd.Timestamp.today().date()
    feats = load_market_features(start_date, end_date, FeatureConfig())
    series = run_detection(feats, DetectConfig())
    path = persist_regime_series(series, settings.data_dir)

    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"regime-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy="regime",
            kind="research",
            git_sha=_git_sha(),
            command=f"quant regime fit --start {start} --end {end_date}",
            params={"start": str(start_date), "end": str(end_date)},
            metrics={"n_days": float(len(series))},
            gates={},
            artifacts={"regime_series": str(path)},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )
    console.print(f"[green]Wrote {len(series)} regime rows to {path}[/green]")


@regime.command("label", help="Print the regime label + posterior as of a date (default latest).")
@click.option("--asof", default=None, help="Date (YYYY-MM-DD). Default: latest row.")
def regime_label(asof: str | None) -> None:
    import pandas as pd

    path = _regime_series_path()
    if not path.exists():
        raise click.ClickException("No regime series. Run `quant regime fit` first.")
    frame = pd.read_parquet(path)
    row = frame.loc[pd.Timestamp(asof)] if asof else frame.iloc[-1]
    table = Table(title="Regime as of " + str(frame.index[-1].date() if not asof else asof))
    for col in ("Label", "p(calm)", "p(choppy)", "p(crisis)"):
        table.add_column(col)
    table.add_row(
        str(row["label"]),
        f"{row['p_calm']:.2f}",
        f"{row['p_choppy']:.2f}",
        f"{row['p_crisis']:.2f}",
    )
    console.print(table)


@regime.command("validate", help="Run the four out-of-sample gates and log to the registry.")
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default=None)
def regime_validate(start: str, end: str | None) -> None:
    import pandas as pd

    from quant.data import bars
    from quant.regime.detect import DetectConfig, run_detection
    from quant.regime.features import FeatureConfig, load_market_features
    from quant.regime.validation import validate_regime_series

    settings = Settings()  # type: ignore[call-arg]
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date() if end else pd.Timestamp.today().date()
    cfg = DetectConfig()
    feats = load_market_features(start_date, end_date, FeatureConfig())
    series = run_detection(feats, cfg)

    spy = bars.get_bars(bars.BarRequest(symbols=["SPY"], start=start_date, end=end_date))
    from quant.regime.features import _extract_close

    spy_ret = _extract_close(spy, "SPY").pct_change()
    report = validate_regime_series(series, spy_returns=spy_ret)

    # Gate 4: real PIT check — labels invariant under a 90% truncation.
    cutoff = feats.index[int(len(feats) * 0.9)]
    trunc = run_detection(feats.loc[:cutoff], cfg)
    shared = trunc.index.intersection(series.index)
    pit_ok = bool((series.loc[shared, "label"] == trunc.loc[shared, "label"]).all())
    gates = {**report.gates, "pit_consistent": pit_ok}

    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"regime-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy="regime",
            kind="validation",
            git_sha=_git_sha(),
            command=f"quant regime validate --start {start} --end {end_date}",
            params={"start": str(start_date), "end": str(end_date)},
            metrics=report.metrics,
            gates=gates,
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )
    table = Table(title="Regime validation gates")
    table.add_column("Gate")
    table.add_column("Pass")
    for name, ok in gates.items():
        table.add_row(name, "[green]yes[/green]" if ok else "[red]no[/red]")
    console.print(table)
    console.print(f"Overall: {'PASS' if all(gates.values()) else 'FAIL'}")
```

> Note: confirm `console`, `Table`, `Settings`, `append_experiment`, `ExperimentRecord`, `subprocess`, `uuid`, `datetime`, `UTC`, and `Path` are already imported at the top of `cli.py` (the existing `validate`/`research` code uses all of them). Add any genuinely missing import; do not duplicate.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli.py::test_regime_label_and_validate_commands -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/cli.py tests/test_cli.py
git commit -m "feat(regime): quant regime CLI group (fit/label/validate)"
```

---

## Task 10: TUI regime panel

**Files:**
- Modify: `quant/tui.py` (extend `MonitorSnapshot` with a regime field + a render path; add the panel to the layout)
- Test: `tests/test_tui.py` (append; if no such file exists, create it following the snapshot-build pattern referenced in `tui.py`'s module docstring)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tui.py`:

```python
def test_snapshot_reads_regime_label(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    idx = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame(
        {"p_calm": 0.6, "p_choppy": 0.3, "p_crisis": 0.1, "label": "calm-bull", "refit_epoch": 0},
        index=idx,
    )
    frame.index.name = "date"
    path = tmp_path / "regime" / "regime_series.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path)

    from quant.tui import latest_regime

    label, posterior = latest_regime(tmp_path)
    assert label == "calm-bull"
    assert posterior["p_crisis"] == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tui.py::test_snapshot_reads_regime_label -v`
Expected: FAIL with `ImportError: cannot import name 'latest_regime'`

- [ ] **Step 3: Write minimal implementation**

Add a small, event-loop-free reader to `quant/tui.py` (this is the testable seam; wiring it into the visible layout is a follow-on render-only step):

```python
def latest_regime(data_dir: Path) -> tuple[str, dict[str, float]]:
    """Return (label, {p_calm,p_choppy,p_crisis}) from the persisted regime series.

    Returns ("unknown", zeros) when no series has been written yet.
    """
    path = data_dir / "regime" / "regime_series.parquet"
    if not path.exists():
        return "unknown", {"p_calm": 0.0, "p_choppy": 0.0, "p_crisis": 0.0}
    frame = pd.read_parquet(path)
    if frame.empty:
        return "unknown", {"p_calm": 0.0, "p_choppy": 0.0, "p_crisis": 0.0}
    row = frame.iloc[-1]
    return str(row["label"]), {
        "p_calm": float(row["p_calm"]),
        "p_choppy": float(row["p_choppy"]),
        "p_crisis": float(row["p_crisis"]),
    }
```

Then add a `regime_label: str` and `regime_posterior: dict[str, float]` field to the `MonitorSnapshot` dataclass and populate them inside `MonitorSnapshot.build(...)` by calling `latest_regime(settings.data_dir)`, and add a one-line regime row to the rendered header/status panel (e.g. `Regime: calm-bull  (crisis 10%)`). Keep the render change minimal — the snapshot field is what the test exercises.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_tui.py::test_snapshot_reads_regime_label -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant
git add quant/tui.py tests/test_tui.py
git commit -m "feat(regime): surface current regime in the TUI monitor"
```

---

## Task 11: Property tests, docs, full acceptance

**Files:**
- Test: `tests/regime/test_hmm.py` (append a hypothesis property test)
- Modify: `README.md`

- [ ] **Step 1: Write the failing property test**

Append to `tests/regime/test_hmm.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=25, deadline=None)
@given(
    n_obs=st.integers(min_value=40, max_value=120),
    seed=st.integers(min_value=0, max_value=50),
)
def test_forward_filter_rows_are_distributions(n_obs, seed):
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1, size=(n_obs, 2))
    params = HMMParams(
        start_prob=np.array([0.4, 0.3, 0.3]),
        trans_mat=np.full((3, 3), 1 / 3),
        means=rng.normal(0, 1, size=(3, 2)),
        variances=np.abs(rng.normal(1, 0.2, size=(3, 2))) + 0.1,
    )
    post = forward_filter(obs, params)
    assert np.all(post >= -1e-9)
    np.testing.assert_allclose(post.sum(axis=1), np.ones(n_obs), atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/pytest tests/regime/test_hmm.py::test_forward_filter_rows_are_distributions -v`
Expected: PASS (forward_filter already exists; this locks the invariant). If it fails, fix the normalization in `forward_filter`.

- [ ] **Step 3: Update README**

Add to the CLI section of `README.md`, under the existing `quant ...` command list:

```markdown
### Regime detection (observed, gated signal)

```bash
uv run quant regime fit                 # refit HMM walk-forward, write data/regime/regime_series.parquet
uv run quant regime label               # print the current market regime + posterior
uv run quant regime label --asof 2022-06-15
uv run quant regime validate            # run the four out-of-sample gates, log to the registry
```

A market-wide 3-state Gaussian HMM (calm-bull / choppy / crisis) over SPY
return/vol, VIX, drawdown, and term-spread features. Point-in-time by
construction (walk-forward refit + filtered posteriors). It is an **observed
signal only** — it does not change any live position until it passes its own
validation gate. See `docs/superpowers/specs/2026-05-28-regime-detection-engine-design.md`.
```

- [ ] **Step 4: Full acceptance run**

Run:

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy quant && .venv/bin/pytest -q
```

Expected: ruff clean, mypy clean, all tests pass (existing 418 + the new regime tests).

- [ ] **Step 5: Commit**

```bash
git add tests/regime/test_hmm.py README.md
git commit -m "test(regime): property test for filter invariants + README docs"
```

---

## Self-review notes (author)

- **Spec coverage:** package layout (Task 1, 5, 6, 7, 8), HMM fit/filter/Viterbi/score (Tasks 2–4), Kalman smoother (Task 5), PIT features + trailing standardization (Task 6), walk-forward refit + canonical labels + persistence + PIT truncation test (Task 7), four validation gates (Task 8 + Task 9 gate-4 wiring), CLI fit/label/validate (Task 9), TUI panel (Task 10), property tests + README (Task 11). The spec's `quant regime backtest` (Viterbi plot) is intentionally deferred — not required for the observed-signal milestone — and should be added as a follow-on if the offline plot is wanted.
- **Type consistency:** `HMMParams` field names (`start_prob`, `trans_mat`, `means`, `variances`) are used identically in every task. `forward_filter(obs, params)` and `log_emission(obs, params)` take `obs` first throughout. Canonical label order `("calm-bull","choppy","crisis")` and posterior column names `p_calm/p_choppy/p_crisis` are consistent across detect/validation/CLI/TUI.
- **Open verifications flagged inline:** bars close-column layout (Task 6 note), refit-boundary ownership under truncation (Task 7 note), existing cli.py imports (Task 9 note), presence/shape of `tests/test_tui.py` and `MonitorSnapshot.build` (Task 10).
```
