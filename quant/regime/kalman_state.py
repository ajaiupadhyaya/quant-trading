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
    obs: np.ndarray = np.asarray(y, dtype=float)
    n = obs.size
    if n == 0:
        result: np.ndarray = obs.copy()
        return result
    out: np.ndarray = np.empty(n)
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
