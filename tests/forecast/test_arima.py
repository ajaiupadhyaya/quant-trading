"""ARIMA conditional-mean modeling (charter techniques) — hand-rolled, pure numpy."""

from __future__ import annotations

import numpy as np

from quant.forecast.arima import (
    ARIMAConfig,
    ARIMAVerdict,
    arima_forecast_next,
    arima_research_verdict,
    fit_arima,
    walk_forward_arima_eval,
)


def _ar1(n: int, seed: int, phi: float = 0.5, sigma: float = 0.01) -> np.ndarray:
    """A genuine AR(1) series: y_t = phi*y_{t-1} + eps."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + sigma * rng.standard_normal()
    return y


def _white(n: int, seed: int, sigma: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sigma * rng.standard_normal(n)


def test_fit_ar1_recovers_coefficient() -> None:
    y = _ar1(3000, seed=1, phi=0.5)
    m = fit_arima(y, ARIMAConfig(p=1, d=0, q=0))
    assert m is not None and len(m.phi) == 1 and m.theta == ()
    assert abs(m.phi[0] - 0.5) < 0.08  # recovered AR(1) coefficient


def test_forecast_correlates_with_truth_on_structured_series() -> None:
    y = _ar1(3000, seed=2, phi=0.5)
    m = fit_arima(y[:2000], ARIMAConfig(p=1, d=0, q=0))
    assert m is not None
    preds, actual = [], []
    for t in range(2000, len(y) - 1):
        f = arima_forecast_next(m, y[: t + 1])
        assert f is not None
        preds.append(f)
        actual.append(y[t + 1])
    ic = float(np.corrcoef(preds, actual)[0, 1])
    assert ic > 0.2  # the machinery detects real AR(1) structure


def test_arma11_fits_and_forecasts_finite() -> None:
    y = _ar1(2000, seed=3, phi=0.4)
    m = fit_arima(y, ARIMAConfig(p=1, d=0, q=1))
    assert m is not None and len(m.phi) == 1 and len(m.theta) == 1
    f = arima_forecast_next(m, y)
    assert f is not None and np.isfinite(f)


def test_white_noise_forecast_is_near_zero() -> None:
    y = _white(2000, seed=4)
    m = fit_arima(y, ARIMAConfig(p=2, d=0, q=1))
    assert m is not None
    f = arima_forecast_next(m, y)
    assert f is not None and abs(f) < 0.01  # no structure -> ~zero forecast


def test_differencing_models_random_walk_increments() -> None:
    # a random walk in levels -> d=1 differences to white noise; fit must succeed.
    rw = np.cumsum(_white(2000, seed=5)) + 100.0
    m = fit_arima(rw, ARIMAConfig(p=1, d=1, q=0))
    assert m is not None and m.d == 1
    f = arima_forecast_next(m, rw)
    assert f is not None and np.isfinite(f)
    assert abs(f - rw[-1]) < 1.0  # next level ~ last level (RW increments ~0)


def test_fit_returns_none_on_short_series() -> None:
    assert fit_arima(np.zeros(100), ARIMAConfig()) is None
    assert fit_arima(np.array([]), ARIMAConfig()) is None


def test_fit_returns_none_on_unsupported_d() -> None:
    assert fit_arima(_ar1(1000, seed=6), ARIMAConfig(p=1, d=2, q=0)) is None


def test_determinism_identical_inputs_identical_model() -> None:
    y = _ar1(1500, seed=7, phi=0.45)
    a = fit_arima(y, ARIMAConfig(p=2, d=0, q=1))
    b = fit_arima(y, ARIMAConfig(p=2, d=0, q=1))
    assert a is not None and b is not None
    assert a.phi == b.phi and a.theta == b.theta and a.mean == b.mean


def test_walk_forward_eval_runs() -> None:
    y = _ar1(1200, seed=8, phi=0.4)
    ev = walk_forward_arima_eval(y, config=ARIMAConfig(p=1, d=0, q=0), min_train=300, refit_every=42)
    assert ev.n_oos > 0
    assert ev.hit_rate is None or 0.0 <= ev.hit_rate <= 1.0
    assert len(ev.oos_strategy_returns) == ev.n_oos


def test_walk_forward_detects_edge_on_strong_ar1() -> None:
    y = _ar1(1600, seed=9, phi=0.5)
    ev = walk_forward_arima_eval(y, config=ARIMAConfig(p=1, d=0, q=0), min_train=300, refit_every=42)
    assert ev.mean_ic is not None and ev.mean_ic > 0.0
    assert ev.hit_rate is not None and ev.hit_rate > 0.5  # better than a coin flip


def test_research_verdict_no_edge_on_white_noise() -> None:
    y = _white(1600, seed=10)
    v = arima_research_verdict(y, d=0, min_train=300, refit_every=42)
    assert isinstance(v, ARIMAVerdict)
    assert v.n_oos > 0
    assert v.deflated_sharpe is not None and v.probabilistic_sharpe is not None
    assert not v.passes  # no edge in noise -> not promotion-eligible (documents EMH)


def test_research_verdict_requires_beating_baseline() -> None:
    # passes requires DSR + PSR AND the point forecast beating the unconditional
    # baseline (mse_ratio < 1) — a cost-free directional tilt alone must not pass.
    y = _white(1600, seed=11)
    v = arima_research_verdict(y, d=0, min_train=300, refit_every=42)
    beats_baseline = v.mse_ratio is not None and v.mse_ratio < 1.0
    assert v.passes is (v.passes_dsr and v.passes_psr and beats_baseline)


def test_research_verdict_passes_on_genuine_ar1_edge() -> None:
    # The gate is a TRUE test: a real, low-turnover, baseline-beating AR(1) edge
    # clears all three bars (so the gate isn't trivially always-False).
    y = _ar1(2500, seed=12, phi=0.5)
    v = arima_research_verdict(y, d=0, min_train=400, refit_every=42)
    assert v.mse_ratio is not None and v.mse_ratio < 1.0
    assert v.passes_dsr and v.passes_psr
    assert v.passes
