# tests/intraday/data/test_aggregate_trades.py
import pandas as pd

from quant.intraday.data.aggregate import trades_to_minute_bars


def _trades():
    idx = pd.to_datetime(
        [
            "2023-06-01T13:30:00Z", "2023-06-01T13:30:20Z", "2023-06-01T13:30:59Z",
            "2023-06-01T13:31:05Z",
        ]
    )
    return pd.DataFrame({"price": [10.0, 11.0, 9.0, 12.0], "size": [100, 200, 100, 50]}, index=idx)


def test_trades_to_minute_bars_ohlcv():
    bars = trades_to_minute_bars(_trades(), symbol="AAPL")
    assert list(bars.index) == list(pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:31:00Z"]))
    first = bars.iloc[0]
    assert first["open"] == 10.0 and first["high"] == 11.0 and first["low"] == 9.0 and first["close"] == 9.0
    assert first["volume"] == 400 and first["trade_count"] == 3
    # vwap = sum(price*size)/sum(size) = (10*100+11*200+9*100)/400 = 10.25
    assert round(first["vwap"], 4) == 10.25


def test_trades_to_minute_bars_empty():
    out = trades_to_minute_bars(pd.DataFrame(columns=["price", "size"]), symbol="AAPL")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
