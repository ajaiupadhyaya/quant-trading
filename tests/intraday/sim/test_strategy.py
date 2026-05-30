import pytest

from quant.intraday.strategy import Order, OrderType, Side


def test_market_order_defaults():
    o = Order(symbol="AAPL", side=Side.BUY, qty=100)
    assert o.type is OrderType.MARKET and o.limit_price is None


def test_limit_order_requires_price():
    with pytest.raises(ValueError):
        Order(symbol="AAPL", side=Side.SELL, qty=10, type=OrderType.LIMIT)


def test_qty_must_be_positive():
    with pytest.raises(ValueError):
        Order(symbol="AAPL", side=Side.BUY, qty=0)
