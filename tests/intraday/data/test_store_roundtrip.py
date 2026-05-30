# tests/intraday/data/test_store_roundtrip.py
from datetime import UTC, date, datetime

import pandas as pd

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.store import MarketDataStore


def _store(tmp_path):
    return MarketDataStore(IntradayConfig(data_root=tmp_path))


def _minute_bars():
    idx = pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:31:00Z"])
    return pd.DataFrame(
        {
            "open": [10, 11],
            "high": [12, 11],
            "low": [9, 10],
            "close": [11, 10],
            "volume": [100, 50],
            "vwap": [10.5, 10.4],
            "trade_count": [3, 2],
        },
        index=idx,
    )


def test_write_then_read_minute_bars(tmp_path):
    store = _store(tmp_path)
    store.write_minute_bars("AAPL", date(2023, 6, 1), _minute_bars())
    got = store.get_minute_bars(
        "AAPL", datetime(2023, 6, 1, tzinfo=UTC), datetime(2023, 6, 2, tzinfo=UTC)
    )
    assert len(got) == 2
    assert got.iloc[0]["open"] == 10 and got.iloc[-1]["close"] == 10


def test_get_minute_bars_missing_returns_empty(tmp_path):
    store = _store(tmp_path)
    got = store.get_minute_bars(
        "ZZZZ", datetime(2023, 6, 1, tzinfo=UTC), datetime(2023, 6, 2, tzinfo=UTC)
    )
    assert got.empty
