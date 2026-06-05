"""Tests for quant.execution.orders."""

from __future__ import annotations

import re
from datetime import date

from quant.execution.orders import OrderSide, OrderTemplate, make_client_order_id


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
