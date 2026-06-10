"""Returns-overlay application of a sizing policy + baseline-vs-sized comparison."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace

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
from quant.forecast.vol import forecast_vol_series
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

    When ``config.vol_source == "forecast"`` the vol-target component is driven by
    the OOS-validated one-day-ahead vol forecast instead of trailing realized vol.
    The forecast series is precomputed ONCE (the model is too costly to refit in
    the per-day loop) and is PIT by construction: ``forecast[t]`` uses ``arr[:t]``,
    matching the ``hist = arr[:t]`` convention. During warm-up (forecast NaN) the
    override is non-finite, so ``compute_gross`` falls back to trailing.
    """
    arr = returns.to_numpy(dtype=float)
    index = returns.index
    n = len(returns)
    vol_fc: np.ndarray | None = None
    if config.use_vol_target and config.vol_source == "forecast":
        vol_fc = forecast_vol_series(
            arr,
            model=config.vol_forecast_model,
            refit_every=config.vol_forecast_refit_every,
            min_obs=config.vol_lookback_days,
        )
    gross_vals = np.empty(n, dtype=float)
    for t in range(n):
        hist = arr[:t]
        prior_ts = index[t - 1] if t > 0 else None
        label = _as_of_label(regime_labels, prior_ts)
        override = float(vol_fc[t]) if vol_fc is not None else None
        gross_vals[t] = compute_gross(hist, label, config, vol_override=override).gross
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


def _vol_tracking(sized: pd.Series, target_vol: float, window: int = 63) -> dict[str, float]:
    """How tightly the SIZED series tracks ``target_vol`` — vol-targeting's actual job.

    Returns the mean and stdev of the rolling annualised realized vol (lower stdev
    = steadier risk) and the mean absolute deviation from target (lower = better
    tracking). The primary gate metric is ``mad_from_target``.
    """
    roll = sized.rolling(window).std(ddof=1) * float(np.sqrt(252))
    roll = roll.dropna()
    if roll.empty:
        return {"roll_vol_mean": 0.0, "roll_vol_std": 0.0, "mad_from_target": 0.0}
    return {
        "roll_vol_mean": float(roll.mean()),
        "roll_vol_std": float(roll.std(ddof=1)),
        "mad_from_target": float((roll - target_vol).abs().mean()),
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


@dataclass(frozen=True)
class VolSourceComparison:
    """Trailing- vs forecast-driven vol-targeting on one return series.

    ``*_metrics`` are the standard return metrics; ``*_tracking`` are the
    vol-targeting metrics (rolling-vol mean/std + MAD-from-target). The honest
    gate: forecast wins only if it TIGHTENS tracking (lower ``mad_from_target``)
    AND does not worsen Sharpe / max-drawdown.
    """

    target_vol: float
    trailing_metrics: dict[str, float]
    forecast_metrics: dict[str, float]
    trailing_tracking: dict[str, float]
    forecast_tracking: dict[str, float]
    trailing_gross_mean: float
    forecast_gross_mean: float


def compare_vol_source(
    returns: pd.Series,
    config: SizingConfig,
    regime_labels: pd.Series | None = None,
    *,
    tracking_window: int = 63,
) -> VolSourceComparison:
    """A/B the vol-target source: trailing realized vol vs the validated forecast.

    Runs the sizing overlay twice on the SAME returns — once with
    ``vol_source="trailing"`` (incumbent), once with ``"forecast"`` — and reports
    return + vol-tracking metrics for each. To isolate the vol-source effect the
    other components are left as configured; pass a config with only vol-target on
    for a clean read. No-lookahead is inherited from ``apply_sizing``.
    """
    trailing_cfg = dc_replace(config, vol_source="trailing")
    forecast_cfg = dc_replace(config, vol_source="forecast")
    sized_t, gross_t = apply_sizing(returns, trailing_cfg, regime_labels)
    sized_f, gross_f = apply_sizing(returns, forecast_cfg, regime_labels)
    return VolSourceComparison(
        target_vol=config.target_vol,
        trailing_metrics=_metrics(sized_t),
        forecast_metrics=_metrics(sized_f),
        trailing_tracking=_vol_tracking(sized_t, config.target_vol, tracking_window),
        forecast_tracking=_vol_tracking(sized_f, config.target_vol, tracking_window),
        trailing_gross_mean=float(gross_t.mean()) if len(gross_t) else 0.0,
        forecast_gross_mean=float(gross_f.mean()) if len(gross_f) else 0.0,
    )
