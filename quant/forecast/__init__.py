"""Forecasting models (roadmap Phase 8) — research-grade, honestly evaluated.

Models here are SHADOW/advisory until they pass out-of-sample evaluation against
strong naive benchmarks; none drives sizing or trading on import. The first is a
HAR-RV volatility forecaster (Corsi 2009) benchmarked against EWMA/RiskMetrics
and a random walk.
"""

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
    "HARModel",
    "VolForecast",
    "compute_vol_forecast",
    "ewma_forecast_series",
    "fit_har",
    "har_forecast_next",
    "live_vol_forecast",
    "qlike",
    "realized_variance",
    "render_vol_forecast",
    "walk_forward_eval",
]
