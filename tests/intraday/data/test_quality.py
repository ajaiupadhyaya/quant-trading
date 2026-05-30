# tests/intraday/data/test_quality.py
from datetime import date

import pandas as pd

from quant.intraday.data.quality import detect_minute_gaps, filter_bad_trades, regular_session_minutes


def test_regular_session_minutes_count():
    # 9:30–16:00 ET = 390 one-minute bars on a normal day
    assert regular_session_minutes(date(2023, 6, 1)) == 390


def test_detect_minute_gaps_finds_missing_minutes():
    idx = pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:32:00Z"])  # 13:31 missing
    bars = pd.DataFrame({"close": [1, 2]}, index=idx)
    gaps = detect_minute_gaps(bars)
    assert pd.Timestamp("2023-06-01T13:31:00Z") in gaps


def test_filter_bad_trades_drops_zero_and_outliers():
    idx = pd.to_datetime(["2023-06-01T13:30:00Z"] * 4)
    df = pd.DataFrame({"price": [100.0, 0.0, -5.0, 1_000_000.0], "size": [10, 10, 10, 10]}, index=idx)
    clean = filter_bad_trades(df, ref_price=100.0, max_deviation=0.2)
    assert list(clean["price"]) == [100.0]  # zero, negative, and 10000x outlier removed
