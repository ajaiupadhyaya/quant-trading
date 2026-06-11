"""Forecast-driven vol-targeting: the default-OFF vol-source bridge + A/B harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.forecast.vol import forecast_vol_series
from quant.sizing.backtest import apply_sizing, compare_vol_source
from quant.sizing.models import SizingConfig
from quant.sizing.policy import compute_gross


def _regime_returns(seed: int = 0) -> np.ndarray:
    """Alternating 100-day low/high vol blocks — a clean vol-regime series."""
    rng = np.random.default_rng(seed)
    blocks = [
        rng.normal(0.0003, (0.08 if k % 2 == 0 else 0.32) / np.sqrt(252), 100) for k in range(10)
    ]
    return np.concatenate(blocks)


# --------------------------------------------------------------------------- #
# forecast_vol_series
# --------------------------------------------------------------------------- #
def test_forecast_vol_series_warmup_is_nan() -> None:
    fc = forecast_vol_series(_regime_returns(), model="gjr", min_obs=252)
    assert np.all(np.isnan(fc[:252]))
    assert np.isfinite(fc[300:]).any()


def test_forecast_vol_series_tracks_regime() -> None:
    r = _regime_returns()
    fc = forecast_vol_series(r, model="har", refit_every=21, min_obs=252)
    # last block is high-vol (~32%); a forecast must read clearly elevated there.
    assert np.nanmean(fc[-50:]) > 0.20


def test_forecast_vol_series_no_lookahead() -> None:
    """forecast[t] must depend only on returns[:t] — mutating the future cannot move it."""
    r = _regime_returns()
    cut = 600
    r2 = r.copy()
    r2[cut:] = r2[cut:] * 5.0 + 0.05  # arbitrary, large perturbation of the future
    a = forecast_vol_series(r, model="gjr", refit_every=21, min_obs=252)
    b = forecast_vol_series(r2, model="gjr", refit_every=21, min_obs=252)
    # forecasts up to and including index `cut` use only returns[:t<=cut] == unchanged
    np.testing.assert_allclose(a[:cut], b[:cut], rtol=0, atol=0, equal_nan=True)


def test_forecast_vol_series_empty() -> None:
    assert forecast_vol_series(np.array([]), model="gjr").size == 0


# --------------------------------------------------------------------------- #
# compute_gross vol_override
# --------------------------------------------------------------------------- #
def test_vol_override_replaces_trailing() -> None:
    cfg = SizingConfig(
        target_vol=0.15, use_kelly=False, use_drawdown=False, use_regime=False, max_leverage=5.0
    )
    hist = _regime_returns()[:300]
    # override 0.30 vol -> scale 0.15/0.30 = 0.5 exactly, independent of trailing.
    d = compute_gross(hist, None, cfg, vol_override=0.30)
    assert d.vol_scale == 0.5
    # None keeps the trailing estimate (different value).
    d0 = compute_gross(hist, None, cfg, vol_override=None)
    assert d0.vol_scale != 0.5


def test_vol_override_ignored_when_non_finite() -> None:
    cfg = SizingConfig(use_kelly=False, use_drawdown=False, use_regime=False)
    hist = _regime_returns()[:300]
    base = compute_gross(hist, None, cfg, vol_override=None)
    for bad in (np.nan, np.inf, 0.0, -0.1):
        assert compute_gross(hist, None, cfg, vol_override=bad).vol_scale == base.vol_scale


# --------------------------------------------------------------------------- #
# apply_sizing forecast path
# --------------------------------------------------------------------------- #
def test_apply_sizing_forecast_differs_from_trailing() -> None:
    r = pd.Series(_regime_returns(), index=pd.bdate_range("2015-01-01", periods=1000))
    cfg_base = SizingConfig(use_kelly=False, use_drawdown=False, use_regime=False)
    _, gross_t = apply_sizing(r, cfg_base, None)  # trailing (default)
    from dataclasses import replace

    _, gross_f = apply_sizing(r, replace(cfg_base, vol_source="forecast"), None)
    # they must diverge somewhere after warm-up
    assert not np.allclose(gross_t.to_numpy()[300:], gross_f.to_numpy()[300:])


def test_apply_sizing_forecast_pit_no_lookahead() -> None:
    """An early gross scalar must not move when a far-future return is perturbed."""
    from dataclasses import replace

    base = _regime_returns()
    idx = pd.bdate_range("2015-01-01", periods=len(base))
    cfg = replace(
        SizingConfig(use_kelly=False, use_drawdown=False, use_regime=False), vol_source="forecast"
    )
    _, g1 = apply_sizing(pd.Series(base, index=idx), cfg, None)
    pert = base.copy()
    pert[700:] += 0.1
    _, g2 = apply_sizing(pd.Series(pert, index=idx), cfg, None)
    np.testing.assert_allclose(g1.to_numpy()[:700], g2.to_numpy()[:700], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# compare_vol_source A/B
# --------------------------------------------------------------------------- #
def test_compare_vol_source_structure_and_tracking() -> None:
    r = pd.Series(_regime_returns(), index=pd.bdate_range("2015-01-01", periods=1000))
    cfg = SizingConfig(
        target_vol=0.15,
        use_kelly=False,
        use_drawdown=False,
        use_regime=False,
        vol_forecast_model="gjr",
    )
    c = compare_vol_source(r, cfg)
    assert c.target_vol == 0.15
    for d in (c.trailing_metrics, c.forecast_metrics):
        assert {"sharpe", "ann_vol", "max_drawdown"} <= set(d)
    for d in (c.trailing_tracking, c.forecast_tracking):
        assert {"roll_vol_mean", "roll_vol_std", "mad_from_target"} <= set(d)
    # On clean alternating vol regimes the forecast should track the target tighter.
    assert c.forecast_tracking["mad_from_target"] < c.trailing_tracking["mad_from_target"]
