"""Windowing, chronological split, and train-only standardization for the DL alpha.
All numpy; NO torch. Carries the no-lookahead discipline: windows use only past values,
the split is chronological (no shuffle), and standardization uses TRAIN statistics only."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_windows(
    series: NDArray[np.float64], window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sliding lagged windows: X[i] = series[i:i+window], y[i] = series[i+window] (next value)."""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("series must be 1-D")
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(s) <= window:
        raise ValueError("series too short for the requested window")
    n = len(s) - window
    X = np.empty((n, window), dtype=np.float64)  # noqa: N806 (matrix convention)
    for i in range(n):
        X[i] = s[i : i + window]
    y = s[window:].copy()
    return X, y


def train_test_split(
    X: NDArray[np.float64],  # noqa: N803 (matrix convention)
    y: NDArray[np.float64],
    train_frac: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Chronological split (no shuffle): first `train_frac` is train, the rest is test."""
    n = len(X)
    cut = int(n * train_frac)
    if cut < 1 or cut >= n:
        raise ValueError("train_frac yields an empty train or test split")
    return X[:cut], y[:cut], X[cut:], y[cut:]


def standardize(
    train: NDArray[np.float64], test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Z-score using TRAIN mean/std only (no lookahead). Returns (train_z, test_z, mu, sd).
    A zero train std is guarded to 1.0 to avoid divide-by-zero."""
    mu = float(np.mean(train))
    sd = float(np.std(train))
    if sd == 0.0:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd
