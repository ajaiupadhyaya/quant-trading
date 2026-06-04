"""Forecasting models (roadmap Phase 8) — research-grade, honestly evaluated.

Models here are SHADOW/advisory until they pass out-of-sample evaluation against
strong naive benchmarks; none drives sizing or trading on import. The first is a
HAR-RV volatility forecaster (Corsi 2009) benchmarked against EWMA/RiskMetrics
and a random walk.
"""

from quant.forecast.factor import (
    FACTOR_UNIVERSE,
    FactorConfig,
    FactorEval,
    FactorScores,
    compute_factor_scores,
    live_factor_scores,
    render_factor_scores,
    walk_forward_factor_eval,
)
from quant.forecast.vol import (
    HARModel,
    VolForecast,
    compute_vol_forecast,
    ewma_forecast_series,
    fit_har,
    har_forecast_next,
    live_vol_forecast,
    qlike,
    realized_variance,
    render_vol_forecast,
    walk_forward_eval,
)

__all__ = [
    "FACTOR_UNIVERSE",
    "FactorConfig",
    "FactorEval",
    "FactorScores",
    "HARModel",
    "VolForecast",
    "compute_factor_scores",
    "compute_vol_forecast",
    "ewma_forecast_series",
    "fit_har",
    "har_forecast_next",
    "live_factor_scores",
    "live_vol_forecast",
    "qlike",
    "realized_variance",
    "render_factor_scores",
    "render_vol_forecast",
    "walk_forward_eval",
    "walk_forward_factor_eval",
]
