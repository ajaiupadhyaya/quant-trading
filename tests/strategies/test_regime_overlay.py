"""Tests for the shared crisis de-risk RegimeOverlay helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies._regime_overlay import RegimeOverlay, RegimeOverlayConfig


def _flat_spy_bars(n: int = 300, price: float = 100.0) -> pd.DataFrame:
    """Build a MultiIndex (symbol, field) bars frame for SPY at a constant price."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    idx.name = "timestamp"
    close = pd.Series(np.full(n, price, dtype=float), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=np.int64),
        },
        index=idx,
    )
    return pd.concat({"SPY": df}, axis=1)


def _stepped_spy_bars(
    n: int = 300,
    break_at: int = 250,
    high_price: float = 100.0,
    low_price: float = 50.0,
) -> pd.DataFrame:
    """SPY held at ``high_price`` then steps down to ``low_price`` at ``break_at``."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    idx.name = "timestamp"
    close = pd.Series(
        np.where(np.arange(n) < break_at, high_price, low_price).astype(float),
        index=idx,
    )
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=np.int64),
        },
        index=idx,
    )
    return pd.concat({"SPY": df}, axis=1)


def test_overlay_neutral_during_calm_market() -> None:
    bars = _flat_spy_bars()
    vix = pd.Series(15.0, index=bars.index)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=RegimeOverlayConfig())
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 1.0


def test_overlay_halves_when_spy_below_200dma() -> None:
    bars = _stepped_spy_bars(n=300, break_at=250, high_price=100.0, low_price=50.0)
    vix = pd.Series(15.0, index=bars.index)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=RegimeOverlayConfig())
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 0.5


def test_overlay_quarters_when_vix_above_30() -> None:
    bars = _flat_spy_bars()
    vix = pd.Series(35.0, index=bars.index)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=RegimeOverlayConfig())
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 0.25


def test_overlay_strategy_equity_break_flattens() -> None:
    bars = _flat_spy_bars()
    vix = pd.Series(15.0, index=bars.index)
    # Strategy equity climbs to 1.5 then collapses to 0.5 — below its own 200dma.
    equity_values = np.where(np.arange(len(bars.index)) < 250, 1.5, 0.5).astype(float)
    strategy_equity = pd.Series(equity_values, index=bars.index)
    config = RegimeOverlayConfig(use_strategy_equity_filter=True)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=config, strategy_equity=strategy_equity)
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 0.0


def test_overlay_disabled_components_return_one() -> None:
    bars = _flat_spy_bars()
    vix = pd.Series(99.0, index=bars.index)
    config = RegimeOverlayConfig(use_spy_filter=False, use_vix_filter=False)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=config)
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 1.0


def test_overlay_bars_columns_are_multiindex() -> None:
    """Sanity check: the helper used in these tests produces the project's wide format."""
    bars = _flat_spy_bars(n=10)
    assert isinstance(bars.columns, pd.MultiIndex)
    assert "SPY" in bars.columns.get_level_values(0)


def test_overlay_clamps_to_unit_interval() -> None:
    """Even with absurd config values the factor stays in [0, 1]."""
    bars = _flat_spy_bars()
    vix = pd.Series(15.0, index=bars.index)
    config = RegimeOverlayConfig(spy_breach_cap=5.0, vix_breach_cap=-2.0)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=config)
    asof = bars.index[-1].date()
    out = overlay.factor(asof)
    assert 0.0 <= out <= 1.0


def test_overlay_skipped_when_history_too_short() -> None:
    """With <200d of bars the SPY filter should bypass (return 1.0 for calm VIX)."""
    bars = _flat_spy_bars(n=50)
    vix = pd.Series(15.0, index=bars.index)
    overlay = RegimeOverlay(bars=bars, vix=vix, config=RegimeOverlayConfig())
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 1.0


def test_overlay_with_none_vix_only_uses_other_components() -> None:
    """If vix=None and use_vix_filter=True, the VIX component should silently skip."""
    bars = _flat_spy_bars()
    overlay = RegimeOverlay(bars=bars, vix=None, config=RegimeOverlayConfig())
    asof = bars.index[-1].date()
    assert overlay.factor(asof) == 1.0
