from __future__ import annotations

import math

import numpy as np

from quant.sizing.components import (
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)


def test_vol_target_scale_basic() -> None:
    # target 0.15, realized 0.30 -> 0.5, well under cap
    assert vol_target_scale(0.30, 0.15, 2.0) == 0.5


def test_vol_target_scale_clamps_to_max() -> None:
    # realized far below target would lever past the cap -> clamp
    assert vol_target_scale(0.01, 0.15, 2.0) == 2.0


def test_vol_target_scale_neutral_on_zero_vol() -> None:
    assert vol_target_scale(0.0, 0.15, 2.0) == 1.0
    assert vol_target_scale(-1.0, 0.15, 2.0) == 1.0
    assert vol_target_scale(float("nan"), 0.15, 2.0) == 1.0


def test_vol_target_scale_monotonic() -> None:
    # higher realized vol -> lower scale
    assert vol_target_scale(0.20, 0.15, 5.0) > vol_target_scale(0.40, 0.15, 5.0)


def test_fractional_kelly_basic() -> None:
    # mu=0.10, var=0.04 -> full kelly 2.5; half -> 1.25; cap 1.0 -> 1.0
    assert fractional_kelly(0.10, 0.04, 0.5, 1.0) == 1.0
    # smaller edge stays under cap: mu=0.01, var=0.04 -> full 0.25, half 0.125
    assert math.isclose(fractional_kelly(0.01, 0.04, 0.5, 1.0), 0.125)


def test_fractional_kelly_negative_edge_is_zero() -> None:
    assert fractional_kelly(-0.05, 0.04, 0.5, 1.0) == 0.0


def test_fractional_kelly_neutral_on_bad_variance() -> None:
    assert fractional_kelly(0.10, 0.0, 0.5, 1.0) == 0.0
    assert fractional_kelly(0.10, -1.0, 0.5, 1.0) == 0.0
    assert fractional_kelly(float("nan"), 0.04, 0.5, 1.0) == 0.0


def test_drawdown_throttle_no_drawdown() -> None:
    # steadily rising equity -> at peak -> factor 1.0
    rets = np.full(300, 0.001)
    assert drawdown_throttle(rets, 0.20) == 1.0


def test_drawdown_throttle_deep_drawdown_floors_to_zero() -> None:
    # +0% then a -25% cumulative crash with dd_floor 0.20 -> 0.0
    rets = np.concatenate([np.zeros(10), np.full(1, -0.25)])
    assert drawdown_throttle(rets, 0.20) == 0.0


def test_drawdown_throttle_partial_ramp() -> None:
    # equity rises to a peak (1.0) then falls 10% from it: 1 + (-0.10)/0.20 = 0.5.
    # The leading 0.0 establishes the peak — matching the repo convention
    # (metrics.max_drawdown / _common.drawdown_leverage_factor) where equity is
    # cumprod(1+returns) with no implicit leading-capital point, so a lone down
    # day is its own peak (dd=0).
    rets = np.array([0.0, -0.10])
    assert math.isclose(drawdown_throttle(rets, 0.20), 0.5)


def test_drawdown_throttle_neutral_on_empty_or_zero_floor() -> None:
    assert drawdown_throttle(np.array([]), 0.20) == 1.0
    assert drawdown_throttle(np.array([-0.5]), 0.0) == 1.0


def test_regime_multiplier_defaults() -> None:
    w = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}
    assert regime_multiplier("calm-bull", w) == 1.0
    assert regime_multiplier("choppy", w) == 0.5
    assert regime_multiplier("crisis", w) == 0.0


def test_regime_multiplier_unknown_and_none() -> None:
    w = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}
    assert regime_multiplier(None, w) == 1.0
    assert regime_multiplier("mystery", w) == 1.0
    assert regime_multiplier("mystery", w, default=0.3) == 0.3
