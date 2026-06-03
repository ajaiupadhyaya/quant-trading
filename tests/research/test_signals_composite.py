"""Composite risk-posture score: renormalization, clamping, labels, sign."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.signals import SignalConfig, build_market_signals, composite_score
from tests.research.conftest import macro_series

CFG = SignalConfig()


def test_composite_all_present_known_value() -> None:
    # Every component = +0.5; convex combination of identical values -> 0.5.
    comps = {k: 0.5 for k, _ in CFG.composite_weights}
    score, label, coverage, n = composite_score(comps, CFG)
    assert score is not None and abs(score - 0.5) < 1e-12
    assert label == "risk-on"
    assert coverage == 1.0
    assert n == 9


def test_composite_renormalizes_over_present() -> None:
    # Only two components present, both +0.5 -> score 0.5 (NOT diluted by missing).
    comps: dict[str, float | None] = {k: None for k, _ in CFG.composite_weights}
    comps["trend"] = 0.5
    comps["realized_vol"] = 0.5
    score, _label, coverage, n = composite_score(comps, CFG)
    assert score is not None and abs(score - 0.5) < 1e-12
    assert n == 2
    assert abs(coverage - 2.0 / 9.0) < 1e-12


def test_composite_all_missing_returns_none() -> None:
    comps = {k: None for k, _ in CFG.composite_weights}
    assert composite_score(comps, CFG) == (None, None, None, 0)


def test_composite_stays_in_range_for_extremes() -> None:
    for v in (1.0, -1.0):
        comps = {k: v for k, _ in CFG.composite_weights}
        score, _, _, _ = composite_score(comps, CFG)
        assert score is not None and -1.0 <= score <= 1.0


def test_composite_label_thresholds_inclusive() -> None:
    # Single present component lets us pin the score exactly to its squashed value.
    def one(val: float) -> str | None:
        comps: dict[str, float | None] = {k: None for k, _ in CFG.composite_weights}
        comps["trend"] = val
        return composite_score(comps, CFG)[1]

    assert one(0.33) == "risk-on"  # boundary inclusive
    assert one(0.329) == "neutral"
    assert one(-0.33) == "risk-off"  # boundary inclusive
    assert one(-0.329) == "neutral"
    assert one(0.0) == "neutral"


def test_composite_sign_convention_calm_beats_stressed() -> None:
    """A calm, rising tape scores higher (more risk-on) than a falling, high-vol one."""
    idx = pd.bdate_range("2022-01-03", periods=640)
    syms = ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "EFA", "EEM"]
    rng = np.random.default_rng(11)

    calm = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.004, len(idx)))) for s in syms},
        index=idx,
    )
    # Stressed: persistent downtrend with ~4x the volatility.
    stressed = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(-0.0008, 0.02, len(idx)))) for s in syms},
        index=idx,
    )
    vix = macro_series(16.0, idx)
    vix_hi = macro_series(34.0, idx)
    d10 = macro_series(4.0, idx)
    d2 = macro_series(4.1, idx)
    asof = idx[-1].date()

    s_calm = build_market_signals(
        closes=calm, vix=vix, dgs10=d10, dgs2=d2, asof=asof
    ).composite_score
    s_stressed = build_market_signals(
        closes=stressed, vix=vix_hi, dgs10=d10, dgs2=d2, asof=asof
    ).composite_score
    assert s_calm is not None and s_stressed is not None
    assert s_calm > s_stressed
