# tests/intraday/data/test_backfill.py
from datetime import date

import pandas as pd

from quant.intraday.data.backfill import backfill_symbol_day
from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.store import MarketDataStore


class FakeHistClient:
    """Stand-in for StockHistoricalDataClient returning fixed frames."""

    def __init__(self):
        self.trade_calls = 0

    def get_trades_df(self, symbol, day):
        self.trade_calls += 1
        return pd.DataFrame(
            {"price": [10.0, 11.0], "size": [100, 200]},
            index=pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:30Z"]),
        )

    def get_quotes_df(self, symbol, day):
        return pd.DataFrame(
            {"bid": [9.9], "ask": [10.1], "bid_size": [5], "ask_size": [5]},
            index=pd.to_datetime(["2023-06-01T13:30:00Z"]),
        )


def test_backfill_writes_all_three_datasets(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path))
    client = FakeHistClient()
    res = backfill_symbol_day(client, store, "AAPL", date(2023, 6, 1))
    assert res.trades_rows == 2 and res.quote_bar_rows == 1 and res.minute_bar_rows == 1
    # partitions queryable through the store
    import datetime as dt

    s, e = dt.datetime(2023, 6, 1, tzinfo=dt.UTC), dt.datetime(2023, 6, 2, tzinfo=dt.UTC)
    assert len(store.get_minute_bars("AAPL", s, e)) == 1


def test_backfill_is_idempotent_and_resumable(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path))
    client = FakeHistClient()
    backfill_symbol_day(client, store, "AAPL", date(2023, 6, 1))
    # second call with skip_existing should NOT re-fetch
    res = backfill_symbol_day(client, store, "AAPL", date(2023, 6, 1), skip_existing=True)
    assert res.skipped is True
    assert client.trade_calls == 1  # not re-fetched
