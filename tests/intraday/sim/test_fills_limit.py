from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar, Trade
from quant.intraday.sim.fills import limit_crosses, limit_fill
from quant.intraday.strategy import Order, OrderType, Side


def _ts():
    return datetime(2023, 6, 1, 13, 30, tzinfo=UTC)


def test_buy_limit_crosses_on_trade_through():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=99.99, size=5)) is True
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=100.01, size=5)) is False


def test_sell_limit_crosses_on_trade_through():
    o = Order("AAPL", Side.SELL, 10, type=OrderType.LIMIT, limit_price=100.0)
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=100.01, size=5)) is True
    assert limit_crosses(o, Trade(_ts(), "AAPL", price=99.99, size=5)) is False


def test_buy_limit_crosses_on_quote_through():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    thru = QuoteBar(_ts(), "AAPL", bid=99.0, ask=99.95, bid_size=1, ask_size=1)
    assert limit_crosses(o, thru) is True


def test_limit_fill_at_limit_price_no_impact():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    f = limit_fill(o, Trade(_ts(), "AAPL", price=99.99, size=5), _ts(), commission_per_share=0.005)
    assert f is not None and f.price == 100.0 and f.impact_cost == 0.0 and f.spread_cost == 0.0
    assert f.commission == 0.05


def test_limit_no_cross_returns_none():
    o = Order("AAPL", Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
    assert (
        limit_fill(o, Trade(_ts(), "AAPL", price=100.5, size=5), _ts(), commission_per_share=0.0)
        is None
    )
