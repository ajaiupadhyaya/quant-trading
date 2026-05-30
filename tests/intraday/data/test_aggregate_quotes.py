# tests/intraday/data/test_aggregate_quotes.py
import pandas as pd

from quant.intraday.data.aggregate import quotes_to_second_bars


def _quotes():
    idx = pd.to_datetime(
        ["2023-06-01T13:30:00.100Z", "2023-06-01T13:30:00.900Z", "2023-06-01T13:30:02.000Z"]
    )
    return pd.DataFrame(
        {
            "bid": [99.0, 99.5, 100.0],
            "ask": [100.0, 100.5, 100.1],
            "bid_size": [3, 4, 5],
            "ask_size": [2, 1, 6],
        },
        index=idx,
    )


def test_quotes_to_second_bars_takes_last_in_second():
    bars = quotes_to_second_bars(_quotes(), symbol="AAPL")
    # 13:30:00 second -> last quote in that second (the .900 one); 13:30:02 -> its own
    assert list(bars.index) == list(
        pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:02Z"])
    )
    assert bars.iloc[0]["bid"] == 99.5 and bars.iloc[0]["ask"] == 100.5
    assert bars.iloc[1]["bid"] == 100.0


def test_quotes_to_second_bars_empty():
    out = quotes_to_second_bars(
        pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size"]), symbol="AAPL"
    )
    assert out.empty
    assert list(out.columns) == ["bid", "ask", "bid_size", "ask_size"]


def test_quotes_to_second_bars_handles_nan_sizes():
    # NaN bid/ask sizes are real in SIP data; must not crash the int cast (regression).
    idx = pd.to_datetime(["2023-06-01T13:30:00Z"])
    df = pd.DataFrame(
        {"bid": [99.0], "ask": [100.0], "bid_size": [float("nan")], "ask_size": [2.0]},
        index=idx,
    )
    out = quotes_to_second_bars(df, symbol="AAPL")
    assert len(out) == 1 and out.iloc[0]["bid"] == 99.0
    assert out.iloc[0]["bid_size"] == 0  # NaN size -> 0, no crash
