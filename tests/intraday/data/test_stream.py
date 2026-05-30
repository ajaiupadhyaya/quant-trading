# tests/intraday/data/test_stream.py
import asyncio
from datetime import datetime, timezone

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.events import QuoteBar
from quant.intraday.data.store import MarketDataStore
from quant.intraday.data.stream import ingest_quotes


async def _fake_source():
    yield {"symbol": "AAPL", "timestamp": datetime(2023, 6, 1, 13, 30, tzinfo=timezone.utc),
           "bid": 1.0, "ask": 1.1, "bid_size": 1, "ask_size": 1}
    yield {"symbol": "AAPL", "timestamp": datetime(2023, 6, 1, 13, 30, 1, tzinfo=timezone.utc),
           "bid": 1.2, "ask": 1.3, "bid_size": 2, "ask_size": 2}


def test_ingest_quotes_pushes_events_to_buffer(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL",)))
    asyncio.run(ingest_quotes(_fake_source(), store))
    live = list(store.subscribe(["AAPL"]))
    assert len(live) == 2
    assert all(isinstance(e, QuoteBar) for e in live)
    assert live[0].bid == 1.0 and live[1].bid == 1.2
