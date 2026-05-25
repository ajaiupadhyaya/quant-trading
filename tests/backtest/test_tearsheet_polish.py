"""Tests for the new tear-sheet charts (rolling Sharpe/vol, underwater, trade P&L)."""

from __future__ import annotations

import base64

import numpy as np
import pandas as pd

from quant.backtest.tearsheet import (
    _rolling_sharpe_chart,
    _rolling_vol_chart,
    _trade_pnl_chart,
    _underwater_chart,
)


def _synthetic_returns(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.012, n), index=idx, name="returns")


def _is_png_base64(s: str) -> bool:
    return s.startswith(("iVBOR", "iVBO")) and len(base64.b64decode(s)) > 100


def test_rolling_sharpe_chart_renders_with_full_history() -> None:
    out = _rolling_sharpe_chart(_synthetic_returns(n=600))
    assert _is_png_base64(out)


def test_rolling_sharpe_chart_handles_short_history() -> None:
    # Less than the 252-day window — should still produce a valid figure.
    out = _rolling_sharpe_chart(_synthetic_returns(n=50))
    assert _is_png_base64(out)


def test_rolling_vol_chart_renders() -> None:
    out = _rolling_vol_chart(_synthetic_returns(n=300))
    assert _is_png_base64(out)


def test_underwater_chart_renders_on_synthetic_equity() -> None:
    rets = _synthetic_returns(n=300)
    equity = (1.0 + rets).cumprod()
    out = _underwater_chart(equity)
    assert _is_png_base64(out)


def test_underwater_chart_handles_empty_input() -> None:
    out = _underwater_chart(pd.Series(dtype=float))
    assert _is_png_base64(out)


def test_trade_pnl_chart_matches_round_trips() -> None:
    """A round-trip set (buy then sell on the same name) should produce a histogram."""
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "qty": 10,
                "side": "buy",
                "fill_price": 100.0,
                "date": pd.Timestamp("2022-01-03"),
            },
            {
                "symbol": "AAA",
                "qty": 10,
                "side": "sell",
                "fill_price": 110.0,
                "date": pd.Timestamp("2022-02-01"),
            },
            {
                "symbol": "BBB",
                "qty": 5,
                "side": "buy",
                "fill_price": 50.0,
                "date": pd.Timestamp("2022-01-04"),
            },
            {
                "symbol": "BBB",
                "qty": 5,
                "side": "sell",
                "fill_price": 45.0,
                "date": pd.Timestamp("2022-02-02"),
            },
        ]
    )
    out = _trade_pnl_chart(trades, pd.Series(dtype=float))
    assert _is_png_base64(out)


def test_trade_pnl_chart_handles_empty_trades() -> None:
    out = _trade_pnl_chart(pd.DataFrame(), pd.Series(dtype=float))
    assert _is_png_base64(out)
