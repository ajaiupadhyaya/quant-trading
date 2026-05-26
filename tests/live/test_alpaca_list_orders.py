"""Tests for AlpacaClient.list_orders()."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from quant.execution.alpaca import AlpacaClient, OrderRow


def _fake_order(
    *,
    client_order_id: str = "trend-20260526-SPY-deadbeef",
    symbol: str = "SPY",
    side: str = "buy",
    qty: str = "69",
    filled_qty: str = "69",
    filled_avg_price: str | None = "500.12",
    submitted_at: datetime = datetime(2026, 5, 26, 19, 55, tzinfo=timezone.utc),  # noqa: UP017
    filled_at: datetime | None = datetime(2026, 5, 26, 19, 55, 4, tzinfo=timezone.utc),  # noqa: UP017
    status: str = "filled",
) -> MagicMock:
    o = MagicMock()
    o.client_order_id = client_order_id
    o.symbol = symbol
    o.side = MagicMock(value=side)
    o.qty = qty
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    o.submitted_at = submitted_at
    o.filled_at = filled_at
    o.status = MagicMock(value=status)
    return o


def test_list_orders_returns_typed_rows() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = [_fake_order()]

    rows = client.list_orders(since=date(2026, 5, 26), until=date(2026, 5, 26))

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, OrderRow)
    assert row.client_order_id == "trend-20260526-SPY-deadbeef"
    assert row.symbol == "SPY"
    assert row.side == "buy"
    assert row.submitted_qty == 69
    assert row.filled_qty == 69
    assert row.filled_avg_price == 500.12
    assert row.status == "filled"


def test_list_orders_handles_unfilled() -> None:
    client = AlpacaClient.__new__(AlpacaClient)
    client._trading = MagicMock()  # type: ignore[attr-defined]
    client._trading.get_orders.return_value = [
        _fake_order(filled_qty="0", filled_avg_price=None, filled_at=None, status="canceled")
    ]

    rows = client.list_orders(since=date(2026, 5, 26), until=date(2026, 5, 26))

    assert rows[0].filled_qty == 0
    assert rows[0].filled_avg_price is None
    assert rows[0].filled_at is None
    assert rows[0].status == "canceled"
