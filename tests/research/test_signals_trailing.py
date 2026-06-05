"""LOOK-AHEAD PROBE (required gate): every signal at T must use only data <= T.

These tests are the contract that makes the engine safe to feed a live decision
layer. If a future bar can change a past signal, they fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.signals import (
    _standardize,
    breadth,
    build_market_signals,
    drawdown,
    momentum,
    realized_vol,
    rsi,
    to_json_dict,
    trend_filter,
)
from tests.research.conftest import close_panel, eq, macro_series

_INTERIOR = [250, 360, 470, 600]


def _macros(index: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    vix = macro_series(18.0, index, slope=0.001)
    d10 = macro_series(4.0, index, slope=0.0005)
    d2 = macro_series(4.2, index, slope=-0.0003)
    return vix, d10, d2


def test_each_signal_family_is_trailing_only() -> None:
    panel = close_panel(seed=5)
    spy = panel["SPY"]
    for t in _INTERIOR:
        ts = panel.index[t]
        # series-returning families on a single price series
        for fn in (lambda s: realized_vol(s), rsi, drawdown, trend_filter):
            full = fn(spy).loc[ts]
            trunc = fn(spy.loc[:ts]).iloc[-1]
            assert eq(float(full), float(trunc)), (fn, t)
        # momentum returns a frame
        mfull = momentum(spy).loc[ts]
        mtrunc = momentum(spy.loc[:ts]).iloc[-1]
        assert all(eq(float(mfull[c]), float(mtrunc[c])) for c in mfull.index)
        # breadth takes the whole panel
        bfull = breadth(panel).loc[ts]
        btrunc = breadth(panel.loc[:ts]).iloc[-1]
        assert eq(float(bfull), float(btrunc))


def test_build_market_signals_row_is_trailing_only() -> None:
    panel = close_panel(seed=6)
    vix, d10, d2 = _macros(panel.index)
    for t in _INTERIOR:
        ts = panel.index[t]
        asof = ts.date()
        full = build_market_signals(closes=panel, vix=vix, dgs10=d10, dgs2=d2, asof=asof)
        trunc = build_market_signals(
            closes=panel.loc[:ts],
            vix=vix.loc[:ts],
            dgs10=d10.loc[:ts],
            dgs2=d2.loc[:ts],
            asof=asof,
        )
        # Internal truncation makes these byte-identical structured outputs.
        assert to_json_dict(full) == to_json_dict(trunc), t


def test_future_rows_do_not_leak() -> None:
    """The case `.iloc[-1]` (without truncation) would fail and `.iloc[loc]` passes:
    a frame that CONTAINS rows after T must yield the same signals as one cut at T."""
    panel = close_panel(seed=7)
    vix, d10, d2 = _macros(panel.index)
    t = 450
    ts = panel.index[t]
    asof = ts.date()
    with_future = build_market_signals(closes=panel, vix=vix, dgs10=d10, dgs2=d2, asof=asof)
    cut = build_market_signals(
        closes=panel.loc[:ts], vix=vix.loc[:ts], dgs10=d10.loc[:ts], dgs2=d2.loc[:ts], asof=asof
    )
    assert to_json_dict(with_future) == to_json_dict(cut)
    # And a wildly different future must not change the as-of read.
    poisoned = panel.copy()
    poisoned.iloc[t + 1 :] = poisoned.iloc[t + 1 :] * 5.0
    poisoned_sig = build_market_signals(closes=poisoned, vix=vix, dgs10=d10, dgs2=d2, asof=asof)
    assert to_json_dict(poisoned_sig) == to_json_dict(cut)


def test_macro_alignment_is_ffill_not_bfill() -> None:
    """A macro print that only exists AFTER T must never appear at an earlier bar."""
    panel = close_panel(seed=8)
    t = 400
    ts = panel.index[t]
    asof = ts.date()
    vix, d10, d2 = _macros(panel.index)
    # Blank VIX on/after T, then drop a spike strictly after T.
    vix_gapped = vix.copy()
    vix_gapped.iloc[t:] = np.nan
    vix_gapped.iloc[t + 5] = 99.0
    a = build_market_signals(closes=panel, vix=vix_gapped, dgs10=d10, dgs2=d2, asof=asof)
    b = build_market_signals(
        closes=panel.loc[:ts],
        vix=vix.copy().loc[:ts].where(vix.loc[:ts].index < ts),
        dgs10=d10.loc[:ts],
        dgs2=d2.loc[:ts],
        asof=asof,
    )
    # The future 99.0 spike must not have leaked into the as-of VIX read.
    assert a.vol is not None
    assert b.vol is not None
    assert eq(a.vol.vix_level, b.vol.vix_level)


def test_standardization_is_rolling_not_full_sample() -> None:
    s = close_panel(seed=9)["SPY"]
    z = _standardize(s, 252, 60)
    # Manual trailing 252-window z at the last point.
    tail = s.iloc[-252:]
    manual = (s.iloc[-1] - tail.mean()) / tail.std(ddof=0)
    assert abs(float(z.iloc[-1]) - float(manual)) < 1e-9
    # A full-sample z would differ materially from the trailing one in general.
    full_z = (s.iloc[-1] - s.mean()) / s.std(ddof=0)
    assert abs(float(z.iloc[-1]) - float(full_z)) > 1e-6


def test_composite_score_is_trailing_only() -> None:
    panel = close_panel(seed=10)
    vix, d10, d2 = _macros(panel.index)
    for t in (300, 500):
        ts = panel.index[t]
        asof = ts.date()
        full = build_market_signals(closes=panel, vix=vix, dgs10=d10, dgs2=d2, asof=asof)
        trunc = build_market_signals(
            closes=panel.loc[:ts],
            vix=vix.loc[:ts],
            dgs10=d10.loc[:ts],
            dgs2=d2.loc[:ts],
            asof=asof,
        )
        assert eq(full.composite_score, trunc.composite_score)
        assert full.composite_label == trunc.composite_label
