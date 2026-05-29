"""Tests for the orphan wind-down helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant.execution.orders import OrderSide
from quant.live.winddown import capped_qty, winddown_orders


def test_cap_binds_when_order_exceeds_participation():
    # ADV $1,000,000, 10% => $100,000 budget; price $100 => 1000 shares max.
    assert capped_qty(5000, 1_000_000.0, 100.0, 0.10) == 1000


def test_cap_passes_through_when_order_within_budget():
    assert capped_qty(200, 1_000_000.0, 100.0, 0.10) == 200


def test_zero_or_negative_adv_is_zero():
    assert capped_qty(500, 0.0, 100.0, 0.10) == 0
    assert capped_qty(500, -1.0, 100.0, 0.10) == 0


def test_nonpositive_price_is_zero():
    assert capped_qty(500, 1_000_000.0, 0.0, 0.10) == 0


def test_nonfinite_inputs_zero():
    assert capped_qty(500, float("nan"), 100.0, 0.10) == 0
    assert capped_qty(500, 1_000_000.0, float("inf"), 0.10) == 0


def test_nonpositive_order_qty_is_zero():
    assert capped_qty(0, 1_000_000.0, 100.0, 0.10) == 0


def _bars(symbol: str, price: float, volume: int, n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame(
        {f: np.full(n, price) for f in ("open", "high", "low", "close")}
        | {"volume": np.full(n, volume, dtype=np.int64)},
        index=idx,
    )
    df.index.name = "timestamp"
    return pd.concat({symbol: df}, axis=1)


def test_long_orphan_generates_sell_only_and_remaining_zero():
    bars = _bars("SPY", 100.0, 50_000_000)  # ADV $5B, cap never binds
    res = winddown_orders("trend", {"SPY": 70}, bars, date(2024, 2, 10), 0.10)
    assert [o.side for o in res.orders] == [OrderSide.SELL]
    assert res.orders[0].qty == 70
    assert res.orders[0].strategy_slug == "trend"
    assert res.remaining["SPY"] == 0
    assert res.skipped == []


def test_short_orphan_generates_buy_to_cover_only():
    bars = _bars("TLT", 90.0, 50_000_000)
    res = winddown_orders("pairs", {"TLT": -40}, bars, date(2024, 2, 10), 0.10)
    assert [o.side for o in res.orders] == [OrderSide.BUY]
    assert res.orders[0].qty == 40
    assert res.remaining["TLT"] == 0


def test_adv_cap_partial_exit_leaves_remaining():
    # ADV = 100*5000 = $500k; 10% => $50k; price 100 => 500 shares max.
    bars = _bars("DBC", 100.0, 5_000)
    res = winddown_orders("trend", {"DBC": 1200}, bars, date(2024, 2, 10), 0.10)
    assert res.orders[0].qty == 500
    assert res.remaining["DBC"] == 700
    assert res.orders[0].side == OrderSide.SELL


def test_symbol_with_no_bars_is_skipped_not_silent():
    bars = _bars("SPY", 100.0, 50_000_000)
    res = winddown_orders("trend", {"ZZZ": 10}, bars, date(2024, 2, 10), 0.10)
    assert res.orders == []
    assert "ZZZ" in res.skipped
    assert res.remaining["ZZZ"] == 10


def test_never_opens_a_new_symbol():
    bars = _bars("SPY", 100.0, 50_000_000)
    res = winddown_orders("trend", {"SPY": 70}, bars, date(2024, 2, 10), 0.10)
    for sym, q in res.remaining.items():
        assert abs(q) <= abs({"SPY": 70}.get(sym, 0))
