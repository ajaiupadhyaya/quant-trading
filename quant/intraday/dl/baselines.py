"""Dependency-free baselines for the DL alpha comparison. NO torch.

- naive (persistence): predict the last in-window value. On the AR-signal series this
  captures autocorrelation; on a return series it is the random-walk-on-returns guess.
- linear: numpy OLS (with intercept) via np.linalg.lstsq."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def naive_predict(X: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N803 (matrix convention)
    """Persistence: predict X[:, -1] (the most recent value in each window)."""
    return np.asarray(X, dtype=np.float64)[:, -1].copy()


def linear_predict(
    X_train: NDArray[np.float64],  # noqa: N803 (matrix convention)
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],  # noqa: N803 (matrix convention)
) -> NDArray[np.float64]:
    """OLS with intercept fit on train, predicted on test (np.linalg.lstsq)."""
    a_train = np.hstack([np.asarray(X_train, dtype=np.float64), np.ones((len(X_train), 1))])
    coef, *_ = np.linalg.lstsq(a_train, np.asarray(y_train, dtype=np.float64), rcond=None)
    a_test = np.hstack([np.asarray(X_test, dtype=np.float64), np.ones((len(X_test), 1))])
    result: NDArray[np.float64] = a_test @ coef
    return result
