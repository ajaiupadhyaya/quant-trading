"""MarketDataStore: the single read/write interface over partitioned Parquet.

Both the backtester (replay) and the live engine (subscribe) go through this
object so they cannot diverge. Reads use DuckDB for fast multi-partition scans;
writes are per (dataset, symbol, date) Parquet partitions.

NOTE: _write sets df.index.name = "index" before writing so that DuckDB's
read_parquet returns the column as "index" (not "__index_level_0__"). The index
comes back tz-aware as America/New_York (pyarrow UTC-stored, pyarrow reads with
local TZ metadata), so _read calls tz_convert("UTC") rather than tz_localize.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from quant.intraday.data.adjustments import Adjustment, adjust_prices
from quant.intraday.data.config import IntradayConfig, partition_path


class MarketDataStore:
    def __init__(self, config: IntradayConfig) -> None:
        self.config = config
        self._adjustments: dict[str, list[Adjustment]] = {}
        self._buffer: list = []

    # ---- write side -------------------------------------------------------
    def _write(self, dataset: str, symbol: str, day: date, df: pd.DataFrame) -> Path:
        path = partition_path(self.config.data_root, dataset, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")  # atomic: never leave a half-written partition
        # Set index name explicitly so DuckDB read_parquet returns column "index"
        # rather than "__index_level_0__" (pandas default when index.name is None).
        df = df.copy()
        df.index.name = "index"
        df.to_parquet(tmp)
        tmp.replace(path)
        return path

    def write_minute_bars(self, symbol: str, day: date, df: pd.DataFrame) -> Path:
        return self._write("minute_bars", symbol, day, df)

    def write_quote_bars(self, symbol: str, day: date, df: pd.DataFrame) -> Path:
        return self._write("quote_bars_1s", symbol, day, df)

    def write_trades(self, symbol: str, day: date, df: pd.DataFrame) -> Path:
        return self._write("trades", symbol, day, df)

    # ---- read side --------------------------------------------------------
    def _read(self, dataset: str, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        glob = str(self.config.data_root / dataset / f"symbol={symbol}" / "date=*.parquet")
        con = duckdb.connect()
        try:
            rel = con.execute(
                'SELECT * FROM read_parquet(?, union_by_name=true) '
                'WHERE "index" >= ? AND "index" < ? ORDER BY "index"',
                [glob, start, end],
            )
            df = rel.df()
        except duckdb.IOException:
            return pd.DataFrame()  # no partitions match the glob
        finally:
            con.close()
        if df.empty:
            return df
        df = df.set_index("index").sort_index()
        # pyarrow stores UTC but reads back with local TZ metadata; convert to UTC
        if df.index.tz is None:
            df.index = pd.DatetimeIndex(df.index, name="timestamp").tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    # ---- adjustment wiring -----------------------------------------------
    def set_adjustments(self, symbol: str, factors: list[Adjustment]) -> None:
        self._adjustments[symbol] = sorted(factors, key=lambda a: a.ex_date)

    def _maybe_adjust(self, symbol: str, df: pd.DataFrame, as_of: date | None) -> pd.DataFrame:
        if as_of is None or df.empty or symbol not in self._adjustments:
            return df
        return adjust_prices(df, self._adjustments[symbol], as_of)

    # ---- public getters --------------------------------------------------
    def get_minute_bars(
        self, symbol: str, start: datetime, end: datetime, as_of: date | None = None
    ) -> pd.DataFrame:
        return self._maybe_adjust(symbol, self._read("minute_bars", symbol, start, end), as_of)

    def get_quote_bars(
        self, symbol: str, start: datetime, end: datetime, as_of: date | None = None
    ) -> pd.DataFrame:
        return self._maybe_adjust(symbol, self._read("quote_bars_1s", symbol, start, end), as_of)

    def get_trades(
        self, symbol: str, start: datetime, end: datetime, as_of: date | None = None
    ) -> pd.DataFrame:
        return self._maybe_adjust(symbol, self._read("trades", symbol, start, end), as_of)

    # ---- event conversion -----------------------------------------------
    def _rows_to_events(self, dataset: str, symbol: str, df: pd.DataFrame):  # type: ignore[return]
        """Convert a DataFrame to typed Event objects matching the dataset schema."""
        from quant.intraday.data.events import Bar, QuoteBar, Trade

        for ts, row in df.iterrows():
            if dataset == "trades":
                yield Trade(
                    ts=ts.to_pydatetime(),
                    symbol=symbol,
                    price=float(row["price"]),
                    size=int(row["size"]),
                )
            elif dataset == "quote_bars_1s":
                yield QuoteBar(
                    ts=ts.to_pydatetime(),
                    symbol=symbol,
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    bid_size=int(row["bid_size"]),
                    ask_size=int(row["ask_size"]),
                )
            elif dataset == "minute_bars":
                yield Bar(
                    ts=ts.to_pydatetime(),
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    vwap=float(row["vwap"]),
                    trade_count=int(row["trade_count"]),
                )

    def replay(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        datasets: tuple[str, ...] = ("minute_bars",),
        as_of: date | None = None,
    ):
        """Yield all events for symbols/datasets in deterministic timestamp order.

        Collects all events then sorts — correct and sufficient for bounded
        backtest windows. Future optimization: heap-merge per-partition iterators
        to avoid full materialization at TB scale.
        """
        from quant.intraday.data.events import event_sort_key

        getter = {
            "trades": self.get_trades,
            "quote_bars_1s": self.get_quote_bars,
            "minute_bars": self.get_minute_bars,
        }
        collected = []
        for symbol in symbols:
            for ds in datasets:
                df = getter[ds](symbol, start, end, as_of=as_of)
                collected.extend(self._rows_to_events(ds, symbol, df))
        collected.sort(key=event_sort_key)
        yield from collected

    # ---- live buffer (subscribe / push / freshness) ----------------------
    def push(self, event) -> None:
        """Append a realtime event to the rolling buffer (called by stream.py)."""
        self._buffer.append(event)

    def subscribe(self, symbols: list[str]):
        """Yield buffered events for symbols in deterministic order.

        Uses the same event_sort_key as replay() — the structural guarantee
        that train-time and serve-time see identical event sequences.
        """
        wanted = set(symbols)
        from quant.intraday.data.events import event_sort_key

        for ev in sorted(self._buffer, key=event_sort_key):
            if ev.symbol in wanted:
                yield ev

    def freshness(self, now: datetime | None = None) -> "Freshness":
        """Return a Freshness snapshot based on the most recent buffered event."""
        if not self._buffer:
            return Freshness(last_event_ts=None)
        last = max(ev.ts for ev in self._buffer)
        return Freshness(last_event_ts=last)


@_dataclass(frozen=True)
class Freshness:
    """Snapshot of the live buffer's staleness relative to a reference time."""

    last_event_ts: datetime | None

    def age_seconds(self, now: datetime) -> float:
        if self.last_event_ts is None:
            return float("inf")
        return (now - self.last_event_ts).total_seconds()

    def is_stale(self, now: datetime, max_age_seconds: float) -> bool:
        return self.age_seconds(now) > max_age_seconds
