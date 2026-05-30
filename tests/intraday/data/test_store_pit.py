# tests/intraday/data/test_store_pit.py
from datetime import UTC, date, datetime

import pandas as pd

from quant.intraday.data.adjustments import Adjustment
from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.store import MarketDataStore


def _bars():
    idx = pd.to_datetime(["2023-05-30T13:30:00Z", "2023-06-02T13:30:00Z"])
    return pd.DataFrame(
        {
            "open": [400.0, 100.0],
            "high": [400.0, 100.0],
            "low": [400.0, 100.0],
            "close": [400.0, 100.0],
            "volume": [1, 1],
            "vwap": [400.0, 100.0],
            "trade_count": [1, 1],
        },
        index=idx,
    )


def test_get_minute_bars_applies_pit_adjustment(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path))
    store.write_minute_bars("AAPL", date(2023, 5, 30), _bars().iloc[[0]])
    store.write_minute_bars("AAPL", date(2023, 6, 2), _bars().iloc[[1]])
    store.set_adjustments(
        "AAPL", [Adjustment(date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)]
    )
    start = datetime(2023, 5, 1, tzinfo=UTC)
    end = datetime(2023, 7, 1, tzinfo=UTC)

    seen_after = store.get_minute_bars("AAPL", start, end, as_of=date(2023, 6, 5))
    assert seen_after.iloc[0]["open"] == 100.0  # split applied

    seen_before = store.get_minute_bars("AAPL", start, end, as_of=date(2023, 5, 31))
    assert seen_before.iloc[0]["open"] == 400.0  # no lookahead
