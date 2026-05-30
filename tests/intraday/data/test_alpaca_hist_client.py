# tests/intraday/data/test_alpaca_hist_client.py
from datetime import date

import pytest


@pytest.mark.alpaca
def test_alpaca_hist_client_returns_normalized_frames():
    from quant.intraday.data.backfill import AlpacaHistClient

    client = AlpacaHistClient()  # reads keys from Settings()
    trades = client.get_trades_df("AAPL", date(2024, 1, 2))
    assert list(trades.columns) == ["price", "size"]
    assert trades.index.tz is not None  # tz-aware UTC
    quotes = client.get_quotes_df("AAPL", date(2024, 1, 2))
    assert set(["bid", "ask", "bid_size", "ask_size"]).issubset(quotes.columns)
