"""Tests for quant.execution.orders."""

from __future__ import annotations

import re
from datetime import date

import pytest

from quant.execution.orders import (
    OrderSide,
    OrderTemplate,
    OrderType,
    TimeInForce,
    make_client_order_id,
)


def test_make_client_order_id_format() -> None:
    coid = make_client_order_id("momentum", "AAPL", date(2026, 5, 23))
    # <slug>-<YYYYMMDD>-<symbol> (deterministic; no uuid suffix)
    assert re.match(r"^momentum-20260523-AAPL$", coid)


def test_make_client_order_id_is_deterministic_across_calls() -> None:
    a = make_client_order_id("momentum", "AAPL", date(2026, 5, 23))
    b = make_client_order_id("momentum", "AAPL", date(2026, 5, 23))
    assert a == b


def test_order_template_round_trip() -> None:
    tpl = OrderTemplate(
        symbol="AAPL",
        qty=10,
        side=OrderSide.BUY,
        strategy_slug="momentum",
    )
    assert tpl.symbol == "AAPL"
    assert tpl.qty == 10
    assert tpl.side is OrderSide.BUY
    assert tpl.strategy_slug == "momentum"


def test_order_template_defaults_reproduce_today() -> None:
    t = OrderTemplate(symbol="SPY", qty=10, side=OrderSide.BUY, strategy_slug="momentum")
    assert t.order_type is OrderType.MARKET
    assert t.time_in_force is TimeInForce.DAY
    assert t.limit_price is None


def test_limit_template_requires_positive_price() -> None:
    t = OrderTemplate(
        symbol="SPY",
        qty=10,
        side=OrderSide.BUY,
        strategy_slug="momentum",
        order_type=OrderType.LIMIT,
        limit_price=420.5,
    )
    assert t.order_type is OrderType.LIMIT
    assert t.limit_price == 420.5


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("inf"), float("nan")])
def test_limit_without_valid_price_raises(bad: float | None) -> None:
    with pytest.raises(ValueError):
        OrderTemplate(
            symbol="SPY",
            qty=10,
            side=OrderSide.BUY,
            strategy_slug="momentum",
            order_type=OrderType.LIMIT,
            limit_price=bad,
        )


def test_market_with_a_limit_price_raises() -> None:
    with pytest.raises(ValueError):
        OrderTemplate(
            symbol="SPY",
            qty=10,
            side=OrderSide.BUY,
            strategy_slug="momentum",
            limit_price=420.5,  # MARKET default + a price is contradictory
        )
