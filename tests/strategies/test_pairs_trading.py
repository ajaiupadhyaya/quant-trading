"""Tests for pairs trading iteration-1 safety knobs (VIX gate, stop-loss)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies._pairs_discovery import PairCandidate
from quant.strategies.pairs_trading import PairsTrading


def test_pairs_default_params_include_iteration_1_knobs() -> None:
    p = PairsTrading.default_params
    assert p["max_active_pairs"] == 3
    assert p["min_half_life"] == 2.0
    assert p["max_half_life"] == 20.0
    assert p["adf_p_max"] == 0.01
    assert p["stop_loss_z"] == 4.5
    assert p["vix_max"] == 25.0


def test_pairs_param_grid_widened() -> None:
    g = PairsTrading.param_grid
    assert 3.0 in g["entry_z"]
    assert 30 in g["lookback_days"]
    assert "stop_loss_z" in g
    assert "vix_max" in g
    assert 3.5 in g["stop_loss_z"]


def test_pairs_vix_gate_returns_empty_when_vix_above_max() -> None:
    """Force-feed a high VIX and assert target_positions returns {} regardless of bars."""
    # Tiny synthetic universe — pairs needs at least SEED_PAIRS members for fallback.
    from quant.strategies.pairs_trading import SEED_PAIRS

    symbols = sorted({s for pair in SEED_PAIRS for s in pair})
    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    idx.name = "timestamp"
    rng = np.random.default_rng(11)
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        close = pd.Series(
            50.0 + i + np.cumsum(rng.normal(0.0, 0.5, len(idx))),
            index=idx,
        )
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
            index=idx,
        )
    bars = pd.concat(frames, axis=1)

    # VIX series at 99 (well above any vix_max default of 25).
    vix = pd.Series(99.0, index=idx, name="vix")

    strat = PairsTrading(
        bars=bars,
        params={"vix_max": 25.0},
        vix=vix,
    )
    pos = strat.target_positions(idx[-1].date(), 200_000.0)
    assert pos == {}, f"expected {{}} when VIX=99 > vix_max=25, got {pos}"


def test_pairs_sizing_is_beta_neutral() -> None:
    """Leg B notional must be |beta| x leg A notional (beta-neutral on the spread),
    not equal-dollar. Inject a known pair with beta=2 and force an entry."""
    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    idx.name = "timestamp"
    close = pd.Series(100.0, index=idx)

    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
            index=idx,
        )

    bars = pd.concat({"AAA": _frame(), "BBB": _frame()}, axis=1)
    vix = pd.Series(10.0, index=idx, name="vix")  # below vix_max -> gate open
    strat = PairsTrading(bars=bars, params={"min_history_days": 50}, vix=vix)

    pc = PairCandidate(
        a="AAA", b="BBB", beta=2.0, alpha=0.0, ar1_rho=0.5,
        half_life_days=10.0, spread_std=0.01, adf_stat=-5.0, adf_passes=True,
    )
    strat._discovered = [pc]
    strat._last_discovery_loc = 10**9  # skip rediscovery, keep the injected pair
    strat._spread_z = lambda pair, loc: -3.0  # z < -entry_z -> long the spread (side +1)

    pos = strat.target_positions(idx[-1].date(), 300_000.0)
    assert "AAA" in pos and "BBB" in pos
    a_notional = abs(pos["AAA"]) * 100.0
    b_notional = abs(pos["BBB"]) * 100.0
    # Beta-neutral: |B notional| ~= |beta| * |A notional| = 2x (was 1x under the bug).
    assert b_notional == pytest.approx(2.0 * a_notional, rel=0.02)
    # Long the spread: long A, short beta*B.
    assert pos["AAA"] > 0 and pos["BBB"] < 0
