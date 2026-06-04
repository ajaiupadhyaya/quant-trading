"""Phase 8 — HAR-RV vol forecasting + honest OOS evaluation."""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from quant.forecast.vol import (
    compute_vol_forecast,
    ewma_forecast_series,
    fit_har,
    har_forecast_next,
    live_vol_forecast,
    log_returns,
    mse,
    qlike,
    realized_variance,
    render_vol_forecast,
    walk_forward_eval,
)

ASOF = date(2026, 6, 4)


def _sim_garch(n: int, seed: int, omega=2e-6, alpha=0.08, beta=0.90) -> np.ndarray:
    """A persistent-volatility close series (GARCH(1,1)) for forecasting tests."""
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - beta)
    r = np.zeros(n)
    close = np.empty(n)
    close[0] = 100.0
    for t in range(1, n):
        s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        r[t] = math.sqrt(s2) * rng.standard_normal()
        close[t] = close[t - 1] * math.exp(r[t])
    return close


def test_log_returns_and_realized_variance() -> None:
    close = np.array([100.0, 101.0, 99.0])
    r = log_returns(close)
    assert r.shape == (2,)
    assert abs(r[0] - math.log(101 / 100)) < 1e-12
    rv = realized_variance(r)
    assert abs(rv[0] - r[0] ** 2) < 1e-18


def test_qlike_zero_at_perfect_and_positive_otherwise() -> None:
    assert abs(qlike(0.0004, 0.0004)) < 1e-12  # forecast == proxy → 0
    assert qlike(0.0002, 0.0004) > 0  # under-forecast penalised
    assert qlike(0.0008, 0.0004) > 0  # over-forecast penalised
    # QLIKE penalises under-forecasting of a spike more than over-forecasting
    assert qlike(0.0002, 0.0004) > qlike(0.0006, 0.0004)


def test_mse() -> None:
    assert abs(mse(0.0003, 0.0004) - (0.0001) ** 2) < 1e-18


def test_ewma_recursion_matches_hand_calc() -> None:
    rv = np.array([1e-4, 4e-4, 2e-4])
    f = ewma_forecast_series(rv, lam=0.94)
    assert abs(f[0] - 1e-4) < 1e-18  # seed
    assert abs(f[1] - (0.94 * 1e-4 + 0.06 * 4e-4)) < 1e-18
    assert abs(f[2] - (0.94 * f[1] + 0.06 * 2e-4)) < 1e-18


def test_fit_har_and_forecast_positive() -> None:
    rv = realized_variance(log_returns(_sim_garch(1500, seed=7)))
    model = fit_har(rv)
    assert model is not None and model.n_obs > 1000
    fc = har_forecast_next(model, rv)
    assert fc is not None and fc > 0 and math.isfinite(fc)


def test_fit_har_returns_none_on_short_series() -> None:
    assert fit_har(np.full(20, 1e-4)) is None


def test_walk_forward_har_beats_random_walk() -> None:
    close = _sim_garch(2500, seed=3)
    ev = walk_forward_eval(close, min_train=400, refit_every=21)
    assert ev.n_oos > 1000
    assert set(ev.scores) >= {"har", "ewma", "rw"}
    # HAR should crush the single-day random walk on QLIKE, and not be the worst.
    assert ev.scores["har"].mean_qlike < ev.scores["rw"].mean_qlike
    assert ev.winner in {"har", "ewma", "rolling"}
    assert ev.dm_stat is not None and ev.dm_pvalue is not None


def test_compute_vol_forecast_uses_har_and_annualizes() -> None:
    f = compute_vol_forecast(_sim_garch(800, seed=1), ASOF, symbol="SPY", vix=16.0)
    assert f.model == "har"
    assert f.forecast_vol_ann is not None and 0.0 < f.forecast_vol_ann < 2.0
    assert f.regime in {"calm", "normal", "elevated", "stressed"}
    assert f.vix is not None and abs(f.vix - 0.16) < 1e-9  # 16.0 normalised to 0.16


def test_compute_vol_forecast_empty_is_unavailable() -> None:
    f = compute_vol_forecast(np.array([]), ASOF)
    assert f.forecast_vol_ann is None and f.regime is None
    assert render_vol_forecast(f) == "Vol forecast: unavailable"


def test_live_vol_forecast_failopen(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant.data import macro as macro_mod

    monkeypatch.setattr(
        macro_mod, "get_series", lambda code: (_ for _ in ()).throw(RuntimeError())
    )
    f = live_vol_forecast(SimpleNamespace(data_dir=tmp_path), ASOF)  # empty cache → no bars
    assert f.forecast_vol_ann is None  # degraded, never raises


def test_render_populated() -> None:
    f = compute_vol_forecast(_sim_garch(800, seed=2), ASOF, symbol="SPY", vix=16.0, oos_skill="beats EWMA")
    out = render_vol_forecast(f)
    assert "SPY HAR 1d-ahead vol=" in out
    assert "regime=" in out
    assert "[beats EWMA]" in out
