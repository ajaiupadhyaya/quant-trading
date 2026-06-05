"""Tests for quant.execution.alpaca."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant.execution.alpaca import AccountInfo, AlpacaClient, PositionRow
from quant.execution.orders import OrderSide, OrderTemplate


@pytest.fixture
def mock_trading_client() -> MagicMock:
    return MagicMock()


def test_account_returns_parsed_info(fake_env: None, mock_trading_client: MagicMock) -> None:
    mock_trading_client.get_account.return_value = MagicMock(
        equity="100000.00",
        last_equity="99500.00",
        buying_power="50000.00",
        pattern_day_trader=False,
        cash="25000.00",
        portfolio_value="100000.00",
    )
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        acct = client.account()
    assert isinstance(acct, AccountInfo)
    assert acct.equity == 100000.00
    assert acct.last_equity == 99500.00
    assert acct.pattern_day_trader is False


def test_positions_returns_list_of_position_rows(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(
            symbol="AAPL",
            qty="10",
            avg_entry_price="180.0",
            market_value="1850.00",
            unrealized_pl="50.00",
            side="long",
            current_price="185.0",
        ),
    ]
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        positions = client.positions()
    assert len(positions) == 1
    assert isinstance(positions[0], PositionRow)
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10
    assert positions[0].avg_entry_price == 180.0


def test_submit_order_includes_client_order_id_prefix(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        client.submit_order(
            OrderTemplate(symbol="AAPL", qty=10, side=OrderSide.BUY, strategy_slug="momentum")
        )
    # The submitted request should include a client_order_id prefixed with the strategy slug
    submitted = mock_trading_client.submit_order.call_args.args[0]
    assert submitted.client_order_id.startswith("momentum-")
    assert "AAPL" in submitted.client_order_id
    assert submitted.qty == 10


def test_submit_order_coid_uses_asof_not_today(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    """The client_order_id must embed the rebalance's `asof` session date, not
    date.today(), so the deterministic-COID idempotency backstop aligns with the
    already_traded_today guard (which queries the broker for `asof`)."""
    from datetime import date

    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        coid = client.submit_order(
            OrderTemplate(symbol="SPY", qty=3, side=OrderSide.BUY, strategy_slug="trend"),
            asof=date(2026, 6, 1),
            dry_run=True,
        )
    assert coid == "trend-20260601-SPY"


def test_submit_order_dry_run_does_not_call_api(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        client.submit_order(
            OrderTemplate(symbol="AAPL", qty=10, side=OrderSide.BUY, strategy_slug="momentum"),
            dry_run=True,
        )
    mock_trading_client.submit_order.assert_not_called()
