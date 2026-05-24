"""Tests for quant.execution.reconciler."""

from __future__ import annotations

from quant.execution.orders import OrderSide
from quant.execution.reconciler import reconcile


def test_no_change_emits_no_orders() -> None:
    orders = reconcile(target={"AAPL": 10}, current={"AAPL": 10}, strategy_slug="momentum")
    assert orders == []


def test_buy_to_open_new_position() -> None:
    orders = reconcile(target={"AAPL": 10}, current={}, strategy_slug="momentum")
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.BUY


def test_sell_to_close_position() -> None:
    orders = reconcile(target={}, current={"AAPL": 10}, strategy_slug="momentum")
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.SELL


def test_resize_upward_emits_buy_delta() -> None:
    orders = reconcile(target={"AAPL": 15}, current={"AAPL": 10}, strategy_slug="momentum")
    assert orders[0].qty == 5
    assert orders[0].side is OrderSide.BUY


def test_resize_downward_emits_sell_delta() -> None:
    orders = reconcile(target={"AAPL": 5}, current={"AAPL": 10}, strategy_slug="momentum")
    assert orders[0].qty == 5
    assert orders[0].side is OrderSide.SELL


def test_flip_long_to_short_emits_two_orders() -> None:
    orders = reconcile(target={"AAPL": -5}, current={"AAPL": 10}, strategy_slug="momentum")
    # First flatten 10 long, then open 5 short.
    assert len(orders) == 2
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.SELL
    assert orders[1].qty == 5
    assert orders[1].side is OrderSide.SELL


def test_flip_short_to_long_emits_two_orders() -> None:
    orders = reconcile(target={"AAPL": 5}, current={"AAPL": -10}, strategy_slug="momentum")
    # First cover 10 short, then open 5 long.
    assert len(orders) == 2
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.BUY
    assert orders[1].qty == 5
    assert orders[1].side is OrderSide.BUY


def test_strategy_slug_propagates() -> None:
    orders = reconcile(target={"AAPL": 10}, current={}, strategy_slug="pairs")
    assert orders[0].strategy_slug == "pairs"
