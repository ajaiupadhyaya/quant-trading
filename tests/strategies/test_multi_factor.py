"""Targeted tests for ``MultiFactor`` regime overlay integration.

The general smoke tests live in ``test_concrete_strategies.py``. This file is
for behavior unique to the strategy's regime-overlay plumbing (Task 1.3 of the
2026-05-25 go-live plan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_multi_factor_overlay_reduces_exposure_when_spy_below_200dma() -> None:
    """Overlay should cut gross notional when SPY is below its 200dma."""
    from quant.strategies.multi_factor import MEGACAP_UNIVERSE, MultiFactor

    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    idx.name = "timestamp"
    rng = np.random.default_rng(7)

    # Strategy universe (megacap) — synthetic random walks, no SPY here.
    frames: dict[str, pd.DataFrame] = {}
    for sym in MEGACAP_UNIVERSE:
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, len(idx)))),
            index=idx,
        )
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
            index=idx,
        )
    bars = pd.concat(frames, axis=1)

    # Separate SPY frame, crashing through 200dma.
    spy_close = pd.Series(np.linspace(450.0, 300.0, len(idx)), index=idx)
    spy_df = pd.DataFrame(
        {"open": spy_close, "high": spy_close, "low": spy_close, "close": spy_close, "volume": 1},
        index=idx,
    )
    spy_bars = pd.concat({"SPY": spy_df}, axis=1)

    vix = pd.Series(15.0, index=idx, name="vix")  # calm VIX, SPY gate dominates

    # Avoid fundamentals/network deps in the test.
    base_params = {
        "use_fundamentals": False,
        "min_history_days": 252,
    }

    strat_off = MultiFactor(
        bars=bars,
        params={**base_params, "regime_overlay_enabled": False},
        vix=vix,
        spy_bars=spy_bars,
    )
    strat_on = MultiFactor(
        bars=bars,
        params={**base_params, "regime_overlay_enabled": True},
        vix=vix,
        spy_bars=spy_bars,
    )
    asof = idx[-1].date()
    pos_off = strat_off.target_positions(asof, 200_000.0)
    pos_on = strat_on.target_positions(asof, 200_000.0)

    if not pos_off:
        # Skip if synthetic data didn't produce any signals — not the path under test.
        import pytest

        pytest.skip("synthetic factor panel produced no picks; cannot assert overlay effect")

    last_close = bars.xs("close", axis=1, level=1).iloc[-1]
    notional_off = sum(
        abs(s) * float(last_close[sym]) for sym, s in pos_off.items() if sym in last_close
    )
    notional_on = sum(
        abs(s) * float(last_close[sym]) for sym, s in pos_on.items() if sym in last_close
    )
    # Factor is 0.5 when SPY below 200dma; allow rounding slack.
    assert notional_on <= notional_off * 0.7, f"on={notional_on:.0f} off={notional_off:.0f}"
