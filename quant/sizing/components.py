"""Pure position-sizing components.

Each function returns a finite float and degrades to a neutral value on bad
input (never raises, never returns NaN) so downstream registry serialization
stays finite. None of these touch I/O or hold state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def vol_target_scale(realized_vol: float, target_vol: float, max_scale: float) -> float:
    """Leverage scalar pushing realized vol toward target, clamped to [0, max_scale].

    Returns 1.0 (neutral) when realized_vol is non-finite or <= 0.
    """
    if not math.isfinite(realized_vol) or realized_vol <= 0.0:
        return 1.0
    if not math.isfinite(target_vol) or target_vol < 0.0:
        return 1.0
    scale = target_vol / realized_vol
    return float(max(0.0, min(max_scale, scale)))


def fractional_kelly(mean_return: float, variance: float, fraction: float, cap: float) -> float:
    """Fractional Kelly fraction f = clamp(fraction * mean/variance, 0, cap).

    Long-only (negative edge -> 0.0). Returns 0.0 when variance <= 0 or any
    input is non-finite.
    """
    if not (math.isfinite(mean_return) and math.isfinite(variance) and math.isfinite(fraction)):
        return 0.0
    if variance <= 0.0:
        return 0.0
    full = mean_return / variance
    scaled = fraction * full
    return float(max(0.0, min(cap, scaled)))


def drawdown_throttle(returns_window: np.ndarray, dd_floor: float) -> float:
    """Daniel-Moskowitz exposure attenuator on a 1-D strategy-equity series.

    Builds trailing equity from returns_window, computes current drawdown vs
    trailing peak, returns the linear ramp 1 + dd/dd_floor clamped to [0, 1].
    Returns 1.0 on empty window or dd_floor <= 0.
    """
    if dd_floor <= 0.0:
        return 1.0
    arr = np.asarray(returns_window, dtype=float)
    if arr.size == 0:
        return 1.0
    arr = np.nan_to_num(arr, nan=0.0)
    equity = np.cumprod(1.0 + arr)
    peak = float(np.maximum.accumulate(equity)[-1])
    current = float(equity[-1])
    if peak <= 0.0:
        return 1.0
    dd = current / peak - 1.0  # non-positive
    if dd >= 0.0:
        return 1.0
    factor = 1.0 + dd / dd_floor
    return float(max(0.0, min(1.0, factor)))


def regime_multiplier(
    label: str | None, weights: Mapping[str, float], default: float = 1.0
) -> float:
    """Map a regime label to an exposure multiplier; unknown/None -> default."""
    if label is None:
        return default
    value = weights.get(label, default)
    if not math.isfinite(value):
        return default
    return float(value)
