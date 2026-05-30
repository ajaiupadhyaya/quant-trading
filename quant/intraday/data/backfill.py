# quant/intraday/data/backfill.py
"""Historical SIP backfill: pull trades+quotes for a symbol/day, aggregate, persist.

Idempotent and resumable: an already-written day is skipped (skip_existing).
The Alpaca client is injected (Protocol) so it is fully unit-testable with a fake."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from quant.intraday.data.aggregate import quotes_to_second_bars, trades_to_minute_bars
from quant.intraday.data.config import partition_path
from quant.intraday.data.quality import filter_bad_trades
from quant.intraday.data.store import MarketDataStore


class HistClient(Protocol):
    def get_trades_df(self, symbol: str, day: date) -> pd.DataFrame: ...
    def get_quotes_df(self, symbol: str, day: date) -> pd.DataFrame: ...


@dataclass(frozen=True)
class BackfillResult:
    symbol: str
    day: date
    trades_rows: int
    quote_bar_rows: int
    minute_bar_rows: int
    skipped: bool = False


def backfill_symbol_day(
    client: HistClient, store: MarketDataStore, symbol: str, day: date, *, skip_existing: bool = False
) -> BackfillResult:
    target = partition_path(store.config.data_root, "minute_bars", symbol, day)
    if skip_existing and target.exists():
        return BackfillResult(symbol, day, 0, 0, 0, skipped=True)

    trades = client.get_trades_df(symbol, day)
    if not trades.empty:
        ref = float(trades["price"].median())
        trades = filter_bad_trades(trades, ref_price=ref, max_deviation=0.5)
    quotes = client.get_quotes_df(symbol, day)

    minute = trades_to_minute_bars(trades, symbol)
    qbars = quotes_to_second_bars(quotes, symbol)

    store.write_trades(symbol, day, trades)
    store.write_quote_bars(symbol, day, qbars)
    store.write_minute_bars(symbol, day, minute)
    return BackfillResult(symbol, day, len(trades), len(qbars), len(minute))


class AlpacaHistClient:
    """Production HistClient backed by alpaca-py's StockHistoricalDataClient (SIP feed)."""

    def __init__(self, settings: object = None) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        from quant.util.config import Settings

        s = settings or Settings()  # type: ignore[call-arg]
        self._client = StockHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)  # type: ignore[union-attr]

    def _day_bounds(self, day: date) -> tuple:
        from datetime import datetime, time, timezone

        start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
        end = datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)
        return start, end

    def get_trades_df(self, symbol: str, day: date) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockTradesRequest

        start, end = self._day_bounds(day)
        req = StockTradesRequest(symbol_or_symbols=symbol, start=start, end=end, feed=DataFeed.SIP)
        raw = self._client.get_stock_trades(req).df
        if raw.empty:
            return pd.DataFrame(columns=["price", "size"])
        df = raw.reset_index()
        df = df.set_index(pd.DatetimeIndex(df["timestamp"]))
        return df[["price", "size"]]

    def get_quotes_df(self, symbol: str, day: date) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest

        start, end = self._day_bounds(day)
        req = StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end, feed=DataFeed.SIP)
        raw = self._client.get_stock_quotes(req).df
        if raw.empty:
            return pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size"])
        df = raw.reset_index().set_index(pd.DatetimeIndex(raw.reset_index()["timestamp"]))
        return df.rename(columns={"bid_price": "bid", "ask_price": "ask"})[
            ["bid", "ask", "bid_size", "ask_size"]
        ]
