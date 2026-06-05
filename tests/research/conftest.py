"""Shared helpers for the quant-signals tests (offline; synthetic data only)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from quant.data.universe import ETF_UNIVERSE
from quant.strategies._common import field_frame
from tests.conftest import synthetic_bars

START = date(2022, 1, 3)
END = date(2024, 6, 28)  # ~640 business days — clears every 200/252 warmup


def eq(a: object, b: object, *, tol: float = 1e-9) -> bool:
    """Scalar equality that treats NaN == NaN as True and uses isclose for floats."""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return bool(a == b)


def close_panel(seed: int = 0, symbols: list[str] | None = None) -> pd.DataFrame:
    """A wide close-price panel over the ETF universe via the shared bar factory."""
    syms = symbols if symbols is not None else list(ETF_UNIVERSE)
    bars = synthetic_bars(syms, START, END, seed=seed)
    return field_frame(bars, "close")


def macro_series(level: float, n_index: pd.Index, *, slope: float = 0.0) -> pd.Series:
    """A smooth synthetic macro series aligned to a price index (FRED-like)."""
    return pd.Series(level + slope * np.arange(len(n_index)), index=n_index)


def rising(n: int = 400, start: float = 100.0, step: float = 0.5) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(start + step * np.arange(n), index=idx)


def falling(n: int = 400, start: float = 300.0, step: float = 0.5) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(start - step * np.arange(n), index=idx)


def constant(n: int = 400, value: float = 100.0) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(np.full(n, value), index=idx)


def peak_then_drop(n: int = 400, drop: float = 0.20) -> pd.Series:
    """Rise to a peak at the midpoint then fall by ``drop`` from that peak."""
    idx = pd.bdate_range("2022-01-03", periods=n)
    half = n // 2
    up = 100.0 + 0.5 * np.arange(half)
    peak = up[-1]
    down = np.linspace(peak, peak * (1.0 - drop), n - half)
    return pd.Series(np.concatenate([up, down]), index=idx)
