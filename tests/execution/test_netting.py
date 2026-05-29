"""Tests for per-symbol order netting."""

from __future__ import annotations

from quant.execution.netting import net_orders
from quant.execution.orders import OrderSide, OrderTemplate


def _o(sym: str, qty: int, side: OrderSide, slug: str) -> OrderTemplate:
    return OrderTemplate(symbol=sym, qty=qty, side=side, strategy_slug=slug)


def test_opposing_buy_sell_nets_to_difference():
    res = net_orders(
        [_o("DBC", 11276, OrderSide.BUY, "defensive"), _o("DBC", 3304, OrderSide.SELL, "trend")]
    )
    assert len(res) == 1
    assert res[0].symbol == "DBC" and res[0].side == OrderSide.BUY and res[0].qty == 11276 - 3304
    assert res[0].strategy_slug == "defensive"  # largest |qty| contributor


def test_equal_opposing_nets_to_zero_dropped():
    assert net_orders([_o("X", 100, OrderSide.BUY, "a"), _o("X", 100, OrderSide.SELL, "b")]) == []


def test_multi_sell_signed_sum_attribution_largest():
    res = net_orders(
        [
            _o("DBC", 100, OrderSide.SELL, "trend"),
            _o("DBC", 455, OrderSide.SELL, "risk-parity"),
            _o("DBC", 1819, OrderSide.SELL, "momentum"),
        ]
    )
    assert len(res) == 1 and res[0].side == OrderSide.SELL and res[0].qty == 100 + 455 + 1819
    assert res[0].strategy_slug == "momentum"  # largest |qty|


def test_nonoverlap_passthrough():
    res = net_orders(
        [_o("BAC", 670, OrderSide.SELL, "multi-factor"), _o("SPY", 5, OrderSide.BUY, "defensive")]
    )
    syms = {o.symbol: (o.side, o.qty, o.strategy_slug) for o in res}
    assert syms["BAC"] == (OrderSide.SELL, 670, "multi-factor")
    assert syms["SPY"] == (OrderSide.BUY, 5, "defensive")


def test_attribution_tie_broken_alphabetically():
    res = net_orders([_o("X", 50, OrderSide.SELL, "zeta"), _o("X", 50, OrderSide.SELL, "alpha")])
    assert res[0].side == OrderSide.SELL and res[0].qty == 100 and res[0].strategy_slug == "alpha"


def test_empty_input():
    assert net_orders([]) == []


def test_output_sorted_by_symbol():
    res = net_orders([_o("ZZZ", 1, OrderSide.BUY, "a"), _o("AAA", 1, OrderSide.BUY, "a")])
    assert [o.symbol for o in res] == ["AAA", "ZZZ"]
