from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.sizing import (
    SizingComparison,
    SizingConfig,
    SizingDecision,
    apply_sizing,
    compare_sizing,
    compute_gross,
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)

_finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


def test_public_api_is_importable() -> None:
    assert SizingConfig and SizingDecision and SizingComparison
    assert compute_gross and apply_sizing and compare_sizing
    assert vol_target_scale and fractional_kelly and drawdown_throttle and regime_multiplier


@given(rv=_finite, tv=st.floats(0.0, 1.0), ms=st.floats(0.0, 10.0))
def test_vol_target_scale_bounded(rv: float, tv: float, ms: float) -> None:
    out = vol_target_scale(rv, tv, ms)
    assert np.isfinite(out)
    assert 0.0 <= out <= max(1.0, ms)


@given(
    arr=st.lists(st.floats(-0.5, 0.5, allow_nan=False), min_size=0, max_size=300),
    floor=st.floats(0.0, 1.0),
)
def test_drawdown_throttle_bounded(arr: list[float], floor: float) -> None:
    out = drawdown_throttle(np.array(arr, dtype=float), floor)
    assert np.isfinite(out)
    assert 0.0 <= out <= 1.0


@settings(max_examples=50)
@given(
    arr=st.lists(st.floats(-0.2, 0.2, allow_nan=False), min_size=0, max_size=400),
    label=st.sampled_from([None, "calm-bull", "choppy", "crisis", "unknown"]),
)
def test_compute_gross_finite_and_capped(arr: list[float], label: str | None) -> None:
    cfg = SizingConfig()
    d = compute_gross(np.array(arr, dtype=float), label, cfg)
    assert np.isfinite(d.gross)
    assert 0.0 <= d.gross <= cfg.max_leverage
