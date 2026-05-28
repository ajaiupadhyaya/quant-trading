"""Point-in-time net beta of book returns vs SPY returns (OLS slope)."""

from __future__ import annotations

import math

import numpy as np

_NEUTRAL = 1.0
_MIN_BETA = 0.0
_MAX_BETA = 3.0


def rolling_beta(book_returns: np.ndarray, spy_returns: np.ndarray) -> float:
    """OLS slope of book on SPY over the supplied (trailing, PIT) window.

    Returns 1.0 (neutral) on degenerate input; clamps result to [0, 3].
    """
    book = np.asarray(book_returns, dtype=float)
    spy = np.asarray(spy_returns, dtype=float)
    n = min(book.size, spy.size)
    if n < 2:
        return _NEUTRAL
    book = book[-n:]
    spy = spy[-n:]
    mask = np.isfinite(book) & np.isfinite(spy)
    if mask.sum() < 2:
        return _NEUTRAL
    book = book[mask]
    spy = spy[mask]
    var = float(np.var(spy, ddof=1))
    if var <= 0.0 or not math.isfinite(var):
        return _NEUTRAL
    cov = float(np.cov(book, spy, ddof=1)[0, 1])
    beta = cov / var
    if not math.isfinite(beta):
        return _NEUTRAL
    return float(max(_MIN_BETA, min(_MAX_BETA, beta)))
