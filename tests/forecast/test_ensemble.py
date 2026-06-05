"""Phase 8 — stacking vol-forecast ensemble + nested purged walk-forward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.forecast.ensemble import (
    BASE_LEARNERS,
    StackConfig,
    StackForecast,
    _verdict,
    build_base_panel,
    compute_stack,
    fit_stacker,
    forward_realized_vol,
    render_stack,
    walk_forward_stack,
)


def _garch_close(seed: int, n: int = 2500) -> pd.Series:
    """Synthetic close with volatility clustering (so vol forecasting is non-trivial)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    vol = np.zeros(n)
    r = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(2e-6 + 0.92 * vol[t - 1] ** 2 + 0.06 * r[t - 1] ** 2)
        r[t] = vol[t] * rng.standard_normal()
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def test_forward_realized_vol_is_strictly_forward() -> None:
    rv = np.full(100, 4e-4)  # constant daily variance
    fwd = forward_realized_vol(rv, h=21)
    # constant var → forward vol = sqrt(252 * var), and the last h are NaN
    assert np.isclose(fwd[0], np.sqrt(252 * 4e-4))
    assert np.isnan(fwd[-1]) and np.isnan(fwd[80])


def test_build_base_panel_columns_and_live_tail() -> None:
    close = _garch_close(1)
    panel = build_base_panel(close, config=StackConfig(min_train=400))
    assert list(panel.columns) == ["rw21", "ewma", "har", "regime", "cp", "target"]
    # vol learners present; recent rows kept with an unknown (NaN) target for live use
    assert panel[["rw21", "ewma", "har"]].notna().all().all()
    assert bool(pd.isna(panel["target"].iloc[-1]))
    # regime/cp default to 0 when not supplied
    assert (panel["regime"] == 0.0).all() and (panel["cp"] == 0.0).all()


def test_fit_stacker_nnls_is_nonnegative() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 0.3, (300, 3))
    y = 0.5 * x[:, 0] + 0.5 * x[:, 2]  # true convex combo of learners 0 and 2
    w = fit_stacker(x, y)
    assert (w >= -1e-9).all()  # non-negative
    assert w[2] > 0.1 and w[0] > 0.1  # loads the informative learners


def test_fit_stacker_ridge_path() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 3))
    y = x @ np.array([1.0, -0.5, 0.25])  # ridge may take a negative weight
    w = fit_stacker(x, y, StackConfig(ridge_lambda=1e-6))
    assert np.allclose(w, [1.0, -0.5, 0.25], atol=1e-2)


def test_walk_forward_stack_runs_and_scores() -> None:
    close = _garch_close(2)
    panel = build_base_panel(close, config=StackConfig(min_train=400))
    ev = walk_forward_stack(panel, StackConfig(min_train=400))
    assert ev.n_oos > 100
    assert ev.best_base in ("rw21", "ewma", "har")
    for k in (*BASE_LEARNERS, "eq3", "stack"):
        assert k in ev.mean_qlike
    assert isinstance(ev.verdict, str) and ev.verdict
    # weights are non-negative (NNLS)
    assert all(v >= -1e-9 for v in ev.avg_weights.values())


def test_compute_stack_predicts_live_row() -> None:
    close = _garch_close(3)
    panel = build_base_panel(close, config=StackConfig(min_train=400))
    sf = compute_stack(panel, StackConfig(min_train=400))
    assert isinstance(sf, StackForecast)
    assert sf.forecast_vol_ann is not None and sf.forecast_vol_ann > 0
    assert set(sf.weights) == set(BASE_LEARNERS)


def test_verdict_branches() -> None:
    base = {"har": 1.0, "rw21": 1.2, "ewma": 1.1, "eq3": 1.05, "stack": 0.95}
    assert "BEATS" in _verdict(base, "har", (-3.0, 0.001))
    assert "NOT DM-significant" in _verdict(base, "har", (-1.0, 0.3))
    worse = {**base, "stack": 1.1}
    assert "does NOT beat" in _verdict(worse, "har", (1.5, 0.2))
    assert "inconclusive" in _verdict({}, None, None)


def test_render_stack() -> None:
    assert render_stack(None) == "Vol ensemble: unavailable"
    f = StackForecast(
        asof="2026-06-04",
        horizon=21,
        forecast_vol_ann=0.142,
        base={c: 0.1 for c in BASE_LEARNERS},
        weights={"har": 0.7, "regime": 0.2, "rw21": 0.0, "ewma": 0.0, "cp": 0.1},
        oos_verdict="research-only",
    )
    out = render_stack(f)
    assert "14.2%" in out and "har=0.70" in out and "research-only" in out
