"""Returns-overlay application of a sizing policy + baseline-vs-sized comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)
from quant.sizing.models import SizingConfig
from quant.sizing.policy import compute_gross
from quant.strategies._common import annualize_vol


def _as_of_label(labels: pd.Series | None, prior_ts: pd.Timestamp | None) -> str | None:
    """Most recent label at or before ``prior_ts`` (yesterday). None if unavailable."""
    if labels is None or prior_ts is None or labels.empty:
        return None
    eligible = labels.loc[:prior_ts]
    if eligible.empty:
        return None
    return str(eligible.iloc[-1])


def apply_sizing(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Apply the sizing overlay. Returns (sized_returns, gross_path), index-aligned.

    For day t, the gross scalar is computed from returns[:t] (strictly prior)
    and the regime label as of t-1 — never today's return or label.
    """
    arr = returns.to_numpy(dtype=float)
    index = returns.index
    n = len(returns)
    gross_vals = np.empty(n, dtype=float)
    for t in range(n):
        hist = arr[:t]
        prior_ts = index[t - 1] if t > 0 else None
        label = _as_of_label(regime_labels, prior_ts)
        gross_vals[t] = compute_gross(hist, label, config).gross
    gross = pd.Series(gross_vals, index=index, name="gross")
    sized = pd.Series(gross_vals * arr, index=index, name="sized_returns")
    return sized, gross


def _metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "ann_vol": annualize_vol(returns),
        "win_rate": win_rate(returns),
    }


@dataclass(frozen=True)
class SizingComparison:
    """Baseline vs sized metrics plus gross-exposure summary."""

    baseline: dict[str, float]
    sized: dict[str, float]
    gross_mean: float
    gross_min: float
    gross_max: float
    config: SizingConfig


def compare_sizing(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
) -> SizingComparison:
    """Compute baseline and sized metrics for ``returns`` under ``config``."""
    sized, gross = apply_sizing(returns, config, regime_labels)
    if len(gross) == 0:
        gmean = gmin = gmax = 0.0
    else:
        gmean = float(gross.mean())
        gmin = float(gross.min())
        gmax = float(gross.max())
    return SizingComparison(
        baseline=_metrics(returns),
        sized=_metrics(sized),
        gross_mean=gmean,
        gross_min=gmin,
        gross_max=gmax,
        config=config,
    )
