import pytest

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.feed import FeedError, LiveQuoteFeed


class _FakeQuote:
    def __init__(self, bid, ask, bs=100, as_=100):
        self.bid_price, self.ask_price = bid, ask
        self.bid_size, self.ask_size = bs, as_


class _FakeDataClient:
    def __init__(self, mapping, raise_exc=None):
        self._mapping, self._raise = mapping, raise_exc

    def get_stock_latest_quote(self, request):
        if self._raise:
            raise self._raise
        return self._mapping


def test_latest_quotes_returns_quotebars():
    client = _FakeDataClient({"QQQ": _FakeQuote(100.0, 100.2)})
    feed = LiveQuoteFeed(symbols=["QQQ"], data_client=client)
    bars = feed.latest_quotes()
    assert len(bars) == 1
    qb = bars[0]
    assert isinstance(qb, QuoteBar)
    assert qb.symbol == "QQQ"
    assert qb.bid == 100.0 and qb.ask == 100.2


def test_feed_error_on_client_exception():
    client = _FakeDataClient({}, raise_exc=RuntimeError("network down"))
    feed = LiveQuoteFeed(symbols=["QQQ"], data_client=client)
    with pytest.raises(FeedError):
        feed.latest_quotes()
