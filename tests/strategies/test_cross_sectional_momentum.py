"""Targeted tests for ``CrossSectionalMomentum`` regime overlay integration.

The general smoke tests live in ``test_concrete_strategies.py``. This file is
for behavior unique to the strategy's regime-overlay plumbing (Task 1.2 of the
2026-05-25 go-live plan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.universe import etf_universe
from quant.strategies.cross_sectional_momentum import CrossSectionalMomentum


def test_momentum_regime_overlay_reduces_exposure_when_spy_below_200dma() -> None:
    """With SPY in a crash regime (close < 200dma), overlay should cut target shares."""
    # Build a synthetic panel: SPY crashes mid-series; other ETFs trend up.
    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    idx.name = "timestamp"
    rng = np.random.default_rng(42)
    frames: dict[str, pd.DataFrame] = {}
    for sym in etf_universe():
        if sym == "SPY":
            # 450 -> 300 over the series; guaranteed below 200dma at end.
            close = pd.Series(np.linspace(450.0, 300.0, len(idx)), index=idx)
        else:
            close = pd.Series(
                100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, len(idx)))),
                index=idx,
            )
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
            index=idx,
        )
    bars = pd.concat(frames, axis=1)

    vix = pd.Series(15.0, index=idx, name="vix")  # calm VIX so SPY filter dominates

    overlay_off = CrossSectionalMomentum(
        bars=bars,
        params={"regime_overlay_enabled": False},
        vix=vix,
    )
    overlay_on = CrossSectionalMomentum(
        bars=bars,
        params={"regime_overlay_enabled": True},
        vix=vix,
    )
    asof = idx[-1].date()
    equity = 200_000.0
    off_pos = overlay_off.target_positions(asof, equity)
    on_pos = overlay_on.target_positions(asof, equity)

    # Quick sanity: both strategies hold something.
    assert sum(off_pos.values()) > 0
    # With overlay on (factor=0.5 since SPY is below its 200dma), the gross
    # notional should be roughly half. Use a 0.7 ceiling to allow for rounding.
    on_notional = sum(abs(s) * float(bars[(sym, "close")].iloc[-1]) for sym, s in on_pos.items())
    off_notional = sum(abs(s) * float(bars[(sym, "close")].iloc[-1]) for sym, s in off_pos.items())
    assert on_notional <= off_notional * 0.7, f"on={on_notional:.0f} off={off_notional:.0f}"
