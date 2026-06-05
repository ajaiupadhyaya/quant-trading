"""Tests for AlpacaClient.list_orders_for_date() — the idempotency-guard hook.

quant.live.rebalance.already_traded_today duck-types this method off the client,
so it must actually exist on AlpacaClient (not just on test fakes) or the
reconcile-then-refuse guard is a no-op against the live broker.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from quant.execution.alpaca import AlpacaClient, OrderRow


def _fake_order() -> MagicMock:
    o = MagicMock()
    o.client_order_id = "trend-20260602-SPY-deadbeef"
    o.symbol = "SPY"
    o.side = MagicMock(value="buy")
    o.qty = "10"
    o.filled_qty = "10"
    o.filled_avg_price = "500.0"
    o.submitted_at = datetime(2026, 6, 2, 19, 55, tzinfo=UTC)
    o.filled_at = datetime(2026, 6, 2, 19, 55, 4, tzinfo=UTC)
    o.status = MagicMock(value="filled")
    return o


def test_list_orders_for_date_returns_rows() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = [_fake_order()]

    rows = client.list_orders_for_date(date(2026, 6, 2))

    assert len(rows) == 1
    assert isinstance(rows[0], OrderRow)
    assert rows[0].symbol == "SPY"


def test_list_orders_for_date_empty_when_none() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = []

    assert client.list_orders_for_date(date(2026, 6, 2)) == []


def test_list_orders_for_date_bounds_query_to_that_day() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = []

    client.list_orders_for_date(date(2026, 6, 2))

    (_, kwargs) = client._trading.get_orders.call_args  # type: ignore[attr-defined]
    req = kwargs["filter"]
    assert req.after == datetime(2026, 6, 2, 0, 0, 0, tzinfo=UTC)
    assert req.until == datetime(2026, 6, 2, 23, 59, 59, 999999, tzinfo=UTC)
