# tests/intraday/data/test_replay.py
from datetime import UTC, date, datetime

import pandas as pd

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.events import Bar, QuoteBar
from quant.intraday.data.store import MarketDataStore


def _seed(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL", "MSFT")))
    day = date(2023, 6, 1)
    qa = pd.DataFrame(
        {"bid": [1.0], "ask": [1.1], "bid_size": [1], "ask_size": [1]},
        index=pd.to_datetime(["2023-06-01T13:30:00Z"]),
    )
    store.write_quote_bars("AAPL", day, qa)
    ba = pd.DataFrame(
        {
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
            "vwap": [1.0],
            "trade_count": [1],
        },
        index=pd.to_datetime(["2023-06-01T13:30:00Z"]),
    )
    store.write_minute_bars("MSFT", day, ba)
    return store


def test_replay_orders_across_symbols_and_types(tmp_path):
    store = _seed(tmp_path)
    events = list(
        store.replay(
            ["AAPL", "MSFT"],
            datetime(2023, 6, 1, tzinfo=UTC),
            datetime(2023, 6, 2, tzinfo=UTC),
            datasets=("quote_bars_1s", "minute_bars"),
        )
    )
    # same ts -> QuoteBar before Bar (event rank); AAPL quote then MSFT bar
    assert [type(e).__name__ for e in events] == ["QuoteBar", "Bar"]
    assert isinstance(events[0], QuoteBar) and events[0].symbol == "AAPL"
    assert isinstance(events[1], Bar) and events[1].symbol == "MSFT"
