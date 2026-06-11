"""Forecast-vol-target overlay: factor math, de-risk-only cap, fail-safe, shadow gate."""

from __future__ import annotations

import numpy as np

from quant.live.voltarget import VolTargetConfig, to_report_dict, voltarget_multiplier


def _series(ann_vol: float, n: int, seed: int = 0, mu: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, ann_vol / np.sqrt(252), n)


def test_insufficient_history_is_safe_noop() -> None:
    r = voltarget_multiplier(_series(0.10, 100), VolTargetConfig(actuate=True))
    assert r.degraded is True
    assert r.multiplier == 1.0 and r.applied == 1.0


def test_rising_forecast_cuts_gross() -> None:
    # calm history then a SHORT, sharp recent vol spike: the one-day-ahead forecast
    # (recent-weighted) jumps above the slower trailing-63d average -> anticipatory cut.
    # (A sustained spike instead just lets trailing catch up — no cut — which is correct.)
    r = np.concatenate([_series(0.07, 460, seed=1), _series(0.90, 6, seed=2)])
    res = voltarget_multiplier(r, VolTargetConfig(actuate=True))
    assert res.degraded is False
    assert res.multiplier < 1.0
    assert res.forecast_vol_ann is not None and res.trailing_vol_ann is not None
    assert res.forecast_vol_ann > res.trailing_vol_ann


def test_shadow_does_not_apply() -> None:
    r = np.concatenate([_series(0.07, 460, seed=1), _series(0.90, 6, seed=2)])
    shadow = voltarget_multiplier(r, VolTargetConfig(actuate=False))
    live = voltarget_multiplier(r, VolTargetConfig(actuate=True))
    assert shadow.multiplier == live.multiplier < 1.0  # same (real) computation
    assert shadow.applied == 1.0  # but shadow applies nothing
    assert live.applied == live.multiplier


def test_derisk_only_never_levers_up() -> None:
    # steady calm: forecast ~ trailing, and even if forecast < trailing the cap is 1.0.
    r = _series(0.10, 500, seed=3, mu=0.0003)
    res = voltarget_multiplier(r, VolTargetConfig(actuate=True))
    assert res.multiplier <= 1.0


def test_floor_clamps_extreme_cut() -> None:
    # an extreme recent spike would push the raw factor below the floor; it's clamped.
    r = np.concatenate([_series(0.05, 460, seed=4), _series(1.20, 4, seed=5)])
    res = voltarget_multiplier(r, VolTargetConfig(actuate=True, floor=0.5))
    assert res.multiplier == 0.5


def test_report_dict_shape() -> None:
    r = _series(0.10, 400, seed=6)
    d = to_report_dict(voltarget_multiplier(r, VolTargetConfig()))
    assert {
        "multiplier",
        "applied",
        "actuated",
        "forecast_vol_ann",
        "trailing_vol_ann",
        "reasons",
        "degraded",
    } <= set(d)
