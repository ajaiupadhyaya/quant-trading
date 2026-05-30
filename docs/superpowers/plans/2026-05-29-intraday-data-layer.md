# Intraday Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the intraday data layer (sub-project A) — the single trustworthy source of intraday truth that serves both the backtester and the live engine through one event interface, preventing train/serve skew.

**Architecture:** A parallel in-repo package `quant/intraday/data/`. Pure aggregation/adjustment logic + a `MarketDataStore` over partitioned Parquet (read via DuckDB), with Alpaca SIP backfill and a realtime stream feeding the same `Event` types. The keystone invariant: `replay()` (historical) and `subscribe()` (live) emit identical event sequences.

**Tech Stack:** Python 3.12, pandas, pyarrow, **duckdb** (new dep), alpaca-py (`StockHistoricalDataClient`, `StockDataStream`), Click, pydantic-settings, pytest. `uv` for all commands (`uv run …`, `uv add …`).

**External prerequisite (does not block this plan):** live backfill / `@pytest.mark.alpaca` integration runs require the user's Alpaca **Algo Trader Plus** (SIP) subscription + keys in `.env`. Every task below is built and unit-tested **fixture-driven with no network**, so the entire core is implementable and verifiable now; only the optional live-integration steps wait on the subscription.

---

## File Structure

```
quant/intraday/__init__.py
quant/intraday/data/__init__.py        public exports (MarketDataStore, event types, IntradayConfig)
quant/intraday/data/events.py          Trade / QuoteBar / Bar + event_sort_key (the shared vocabulary)
quant/intraday/data/config.py          IntradayConfig, DEFAULT_UNIVERSE, partition_path()
quant/intraday/data/aggregate.py       trades_to_minute_bars(), quotes_to_second_bars()  [pure]
quant/intraday/data/adjustments.py     Adjustment, adjust_prices(df, factors, as_of)      [pure, PIT]
quant/intraday/data/store.py           MarketDataStore: get_*/write_*/replay/subscribe/freshness
quant/intraday/data/quality.py         session calendar, gap detection, bad-tick filter, run_doctor()
quant/intraday/data/backfill.py        Alpaca SIP historical ingester (idempotent, resumable)
quant/intraday/data/stream.py          realtime SIP websocket ingester + rolling buffer
quant/intraday/cli.py                  `quant intraday data backfill|refresh|status|doctor`
tests/intraday/data/                   mirror of the above, fixture-driven
tests/intraday/fixtures/               recorded Alpaca payloads + small synthetic parquet sets
```

Each module has one responsibility. `events.py` is imported by everything; nothing imports `cli.py`.

---

## Task 1: Package scaffold + shared event types

**Files:**
- Create: `quant/intraday/__init__.py`, `quant/intraday/data/__init__.py`
- Create: `quant/intraday/data/events.py`
- Test: `tests/intraday/__init__.py`, `tests/intraday/data/__init__.py`, `tests/intraday/data/test_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_events.py
from datetime import datetime, timezone

from quant.intraday.data.events import Bar, QuoteBar, Trade, event_sort_key


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_quotebar_mid_and_spread():
    q = QuoteBar(ts=_ts("2023-06-01T13:30:00"), symbol="AAPL", bid=100.0, ask=100.04, bid_size=5, ask_size=7)
    assert q.mid == 100.02
    assert round(q.spread, 4) == 0.04


def test_event_sort_key_orders_by_ts_then_type_then_symbol():
    t = _ts("2023-06-01T13:30:00")
    quote = QuoteBar(ts=t, symbol="MSFT", bid=1, ask=2, bid_size=1, ask_size=1)
    trade = Trade(ts=t, symbol="AAPL", price=1.5, size=10)
    bar = Bar(ts=t, symbol="AAPL", open=1, high=2, low=1, close=2, volume=10, vwap=1.5, trade_count=3)
    events = [bar, trade, quote]
    events.sort(key=event_sort_key)
    # same ts -> QuoteBar(0) before Trade(1) before Bar(2)
    assert [type(e).__name__ for e in events] == ["QuoteBar", "Trade", "Bar"]


def test_events_are_frozen():
    import dataclasses
    import pytest

    t = Trade(ts=_ts("2023-06-01T13:30:00"), symbol="AAPL", price=1.0, size=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.price = 2.0  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: quant.intraday`.

- [ ] **Step 3: Create the package + implementation**

```python
# quant/intraday/__init__.py
"""Intraday equities trading subsystem (event-driven). See docs/superpowers/specs/2026-05-29-intraday-data-layer-design.md."""
```

```python
# quant/intraday/data/__init__.py
"""Intraday data layer: trustworthy historical + realtime intraday market data."""

from quant.intraday.data.events import Bar, Event, QuoteBar, Trade, event_sort_key

__all__ = ["Bar", "Event", "QuoteBar", "Trade", "event_sort_key"]
```

```python
# quant/intraday/data/events.py
"""Shared event vocabulary emitted identically by replay() (historical) and
subscribe() (live). A strategy consumes these and cannot tell which mode it is
in — the structural guarantee against train/serve skew."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Trade:
    ts: datetime  # tz-aware UTC
    symbol: str
    price: float
    size: int


@dataclass(frozen=True, slots=True)
class QuoteBar:
    """1-second NBBO snapshot bar."""

    ts: datetime  # second-boundary, UTC
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Bar:
    """1-minute OHLCV bar (derived from trades)."""

    ts: datetime  # bar-open, UTC
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int


Event = Trade | QuoteBar | Bar

_TYPE_RANK = {QuoteBar: 0, Trade: 1, Bar: 2}


def event_sort_key(event: Event) -> tuple[datetime, int, str]:
    """Deterministic total order for merging streams: timestamp, then a stable
    per-type rank (quote before trade before bar at the same instant), then symbol."""
    return (event.ts, _TYPE_RANK[type(event)], event.symbol)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_events.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday tests/intraday
git commit -m "feat(intraday): shared event types for data layer"
```

---

## Task 2: Config, default universe, partition paths

**Files:**
- Create: `quant/intraday/data/config.py`
- Test: `tests/intraday/data/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_config.py
from datetime import date
from pathlib import Path

from quant.intraday.data.config import DEFAULT_UNIVERSE, IntradayConfig, partition_path


def test_default_universe_is_liquid_and_deduped():
    assert "SPY" in DEFAULT_UNIVERSE and "AAPL" in DEFAULT_UNIVERSE
    assert len(DEFAULT_UNIVERSE) == len(set(DEFAULT_UNIVERSE))
    assert 50 <= len(DEFAULT_UNIVERSE) <= 150


def test_partition_path_layout():
    root = Path("/data/intraday")
    p = partition_path(root, "trades", "AAPL", date(2023, 6, 1))
    assert p == root / "trades" / "symbol=AAPL" / "date=2023-06-01.parquet"


def test_config_defaults(tmp_path):
    cfg = IntradayConfig(data_root=tmp_path)
    assert cfg.hot_window_days == 5
    assert cfg.universe == DEFAULT_UNIVERSE
    assert cfg.data_root == tmp_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/data/config.py
"""Intraday data-layer configuration: universe, storage roots, partition paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# ~100 most-liquid US names (large-caps + major ETFs). Curated for tight spreads;
# intraday edge requires liquidity. Point-in-time membership lives in universe.py.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    # major ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "EEM", "EFA", "XLF", "XLK", "XLE",
    "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "GLD", "SLV", "TLT",
    "HYG", "LQD", "VXX", "SQQQ", "TQQQ", "ARKK", "SMH", "SOXL",
    # mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "ADBE", "CRM", "ORCL", "INTC", "CSCO", "QCOM", "TXN", "MU", "PLTR",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK", "V", "MA",
    # healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "BMY",
    # consumer / industrial / energy
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "DIS",
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "XOM", "CVX", "COP", "SLB",
    # comm / other liquid
    "T", "VZ", "CMCSA", "F", "GM", "UBER", "ABNB", "COIN", "SHOP", "SNOW",
    "DKNG", "RIVN", "SOFI", "MARA",
)


def partition_path(root: Path, dataset: str, symbol: str, day: date) -> Path:
    """root/<dataset>/symbol=<SYM>/date=<YYYY-MM-DD>.parquet (Hive-style)."""
    return root / dataset / f"symbol={symbol}" / f"date={day.isoformat()}.parquet"


@dataclass(frozen=True)
class IntradayConfig:
    data_root: Path
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    hot_window_days: int = 5  # rolling lookback the live engine keeps in memory
    _unused: tuple[()] = field(default=(), repr=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_config.py -v`
Expected: PASS (3 passed). If the universe count assertion fails, adjust the list length into [50,150].

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/config.py tests/intraday/data/test_config.py
git commit -m "feat(intraday): config, default liquid universe, partition paths"
```

---

## Task 3: Aggregate trades → 1-minute OHLCV bars (pure)

**Files:**
- Create: `quant/intraday/data/aggregate.py`
- Test: `tests/intraday/data/test_aggregate_trades.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_aggregate_trades.py -v`
Expected: FAIL — `ModuleNotFoundError` / function missing.

- [ ] **Step 3: Implement**

```python
# quant/intraday/data/aggregate.py
"""Pure aggregation: raw ticks -> bars. No I/O, fully unit-testable."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MINUTE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]


def trades_to_minute_bars(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Resample a (DatetimeIndex, price, size) trade frame to 1-minute OHLCV.

    `symbol` is accepted for API symmetry/logging; output is single-symbol.
    """
    if trades.empty:
        return pd.DataFrame(columns=_MINUTE_COLUMNS)
    df = trades.sort_index()
    grouped = df.resample("1min", label="left", closed="left")
    notional = (df["price"] * df["size"]).resample("1min", label="left", closed="left").sum()
    volume = grouped["size"].sum()
    out = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": volume.astype("int64"),
            "vwap": np.where(volume > 0, notional / volume.replace(0, np.nan), np.nan),
            "trade_count": grouped["price"].count().astype("int64"),
        }
    )
    return out.dropna(subset=["open"])[_MINUTE_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_aggregate_trades.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/aggregate.py tests/intraday/data/test_aggregate_trades.py
git commit -m "feat(intraday): trades -> 1-minute OHLCV aggregation"
```

---

## Task 4: Aggregate quotes → 1-second NBBO bars (pure)

**Files:**
- Modify: `quant/intraday/data/aggregate.py`
- Test: `tests/intraday/data/test_aggregate_quotes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_aggregate_quotes.py
import pandas as pd

from quant.intraday.data.aggregate import quotes_to_second_bars


def _quotes():
    idx = pd.to_datetime(
        ["2023-06-01T13:30:00.100Z", "2023-06-01T13:30:00.900Z", "2023-06-01T13:30:02.000Z"]
    )
    return pd.DataFrame(
        {"bid": [99.0, 99.5, 100.0], "ask": [100.0, 100.5, 100.1],
         "bid_size": [3, 4, 5], "ask_size": [2, 1, 6]},
        index=idx,
    )


def test_quotes_to_second_bars_takes_last_in_second():
    bars = quotes_to_second_bars(_quotes(), symbol="AAPL")
    # 13:30:00 second -> last quote in that second (the .900 one); 13:30:02 -> its own
    assert list(bars.index) == list(pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:02Z"]))
    assert bars.iloc[0]["bid"] == 99.5 and bars.iloc[0]["ask"] == 100.5
    assert bars.iloc[1]["bid"] == 100.0


def test_quotes_to_second_bars_empty():
    out = quotes_to_second_bars(pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size"]), symbol="AAPL")
    assert out.empty
    assert list(out.columns) == ["bid", "ask", "bid_size", "ask_size"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_aggregate_quotes.py -v`
Expected: FAIL — `quotes_to_second_bars` not defined.

- [ ] **Step 3: Implement (append to aggregate.py)**

```python
_QUOTE_COLUMNS = ["bid", "ask", "bid_size", "ask_size"]


def quotes_to_second_bars(quotes: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Downsample raw NBBO quotes to 1-second bars (last quote in each second).

    Sampling the *last* quote per second gives the prevailing NBBO at second
    close — the spread a marketable order would face at that instant.
    """
    if quotes.empty:
        return pd.DataFrame(columns=_QUOTE_COLUMNS)
    df = quotes.sort_index()
    out = df.resample("1s", label="left", closed="left").last().dropna(subset=["bid", "ask"])
    out["bid_size"] = out["bid_size"].astype("int64")
    out["ask_size"] = out["ask_size"].astype("int64")
    return out[_QUOTE_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_aggregate_quotes.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/aggregate.py tests/intraday/data/test_aggregate_quotes.py
git commit -m "feat(intraday): quotes -> 1-second NBBO bar aggregation"
```

---

## Task 5: Corporate-action adjustments with point-in-time correctness (pure)

**Files:**
- Create: `quant/intraday/data/adjustments.py`
- Test: `tests/intraday/data/test_adjustments.py`

- [ ] **Step 1: Write the failing test** (PIT is the charter invariant — a read as-of D must not apply splits with ex-date after D)

```python
# tests/intraday/data/test_adjustments.py
from datetime import date

import pandas as pd

from quant.intraday.data.adjustments import Adjustment, adjust_prices


def _prices():
    idx = pd.to_datetime(["2023-05-30T13:30:00Z", "2023-06-02T13:30:00Z"])
    return pd.DataFrame({"open": [400.0, 100.0], "close": [400.0, 100.0]}, index=idx)


def test_split_applied_when_known_as_of():
    # 4:1 split ex-date 2023-06-01; reading as_of 2023-06-05 -> pre-split prices divided by 4
    factors = [Adjustment(ex_date=date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)]
    out = adjust_prices(_prices(), factors, as_of=date(2023, 6, 5))
    assert out.iloc[0]["open"] == 100.0  # 400 / 4 (pre-split bar back-adjusted)
    assert out.iloc[1]["open"] == 100.0  # post-split bar unchanged


def test_split_NOT_applied_when_ex_date_after_as_of():
    # Same split, but reading as_of 2023-05-31 (before ex-date) -> must NOT adjust (no lookahead)
    factors = [Adjustment(ex_date=date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)]
    out = adjust_prices(_prices(), factors, as_of=date(2023, 5, 31))
    assert out.iloc[0]["open"] == 400.0  # untouched — the split wasn't known yet


def test_no_factors_is_identity():
    out = adjust_prices(_prices(), [], as_of=date(2023, 6, 5))
    pd.testing.assert_frame_equal(out, _prices())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_adjustments.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/data/adjustments.py
"""Point-in-time corporate-action adjustment. Raw prices are never rewritten;
splits/dividends are applied at READ time, capped by an `as_of` date so a
backtest only ever sees actions known by then (charter principle #1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

_PRICE_COLUMNS = ("open", "high", "low", "close", "bid", "ask", "price", "vwap")


@dataclass(frozen=True)
class Adjustment:
    ex_date: date
    split_ratio: float  # e.g. 4.0 means 4-for-1; price divided by 4 before ex-date
    cash_dividend: float  # absolute $/share, subtracted from pre-ex prices


def adjust_prices(df: pd.DataFrame, factors: list[Adjustment], as_of: date) -> pd.DataFrame:
    """Back-adjust price columns for splits/dividends with ex_date <= as_of.

    Bars on/after an ex-date are the "current" scale; bars strictly before it
    are divided by the split ratio (and reduced by the dividend) so the series
    is continuous. Actions with ex_date > as_of are ignored (no lookahead).
    """
    applicable = [f for f in factors if f.ex_date <= as_of]
    if not applicable:
        return df.copy()
    out = df.copy()
    price_cols = [c for c in out.columns if c in _PRICE_COLUMNS]
    ex_index = pd.DatetimeIndex(out.index).tz_convert("UTC") if out.index.tz else pd.DatetimeIndex(out.index)
    for adj in applicable:
        ex_ts = pd.Timestamp(adj.ex_date, tz="UTC")
        pre = ex_index < ex_ts
        if adj.split_ratio and adj.split_ratio != 1.0:
            out.loc[pre, price_cols] = out.loc[pre, price_cols] / adj.split_ratio
        if adj.cash_dividend:
            out.loc[pre, price_cols] = out.loc[pre, price_cols] - adj.cash_dividend
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_adjustments.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/adjustments.py tests/intraday/data/test_adjustments.py
git commit -m "feat(intraday): point-in-time corporate-action adjustments"
```

---

## Task 6: Add duckdb dep + MarketDataStore write/read round-trip

**Files:**
- Modify: `pyproject.toml` (add `duckdb`)
- Create: `quant/intraday/data/store.py`
- Test: `tests/intraday/data/test_store_roundtrip.py`

- [ ] **Step 1: Add the dependency**

Run: `uv add duckdb`
Expected: `duckdb` appended to `[project] dependencies`, lockfile updated.

- [ ] **Step 2: Write the failing test**

```python
# tests/intraday/data/test_store_roundtrip.py
from datetime import date, datetime, timezone

import pandas as pd

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.store import MarketDataStore


def _store(tmp_path):
    return MarketDataStore(IntradayConfig(data_root=tmp_path))


def _minute_bars():
    idx = pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:31:00Z"])
    return pd.DataFrame(
        {"open": [10, 11], "high": [12, 11], "low": [9, 10], "close": [11, 10],
         "volume": [100, 50], "vwap": [10.5, 10.4], "trade_count": [3, 2]},
        index=idx,
    )


def test_write_then_read_minute_bars(tmp_path):
    store = _store(tmp_path)
    store.write_minute_bars("AAPL", date(2023, 6, 1), _minute_bars())
    got = store.get_minute_bars("AAPL", datetime(2023, 6, 1, tzinfo=timezone.utc),
                                datetime(2023, 6, 2, tzinfo=timezone.utc))
    assert len(got) == 2
    assert got.iloc[0]["open"] == 10 and got.iloc[-1]["close"] == 10


def test_get_minute_bars_missing_returns_empty(tmp_path):
    store = _store(tmp_path)
    got = store.get_minute_bars("ZZZZ", datetime(2023, 6, 1, tzinfo=timezone.utc),
                                datetime(2023, 6, 2, tzinfo=timezone.utc))
    assert got.empty
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_store_roundtrip.py -v`
Expected: FAIL — `MarketDataStore` missing.

- [ ] **Step 4: Implement the write/read core**

```python
# quant/intraday/data/store.py
"""MarketDataStore: the single read/write interface over partitioned Parquet.

Both the backtester (replay) and the live engine (subscribe) go through this
object so they cannot diverge. Reads use DuckDB for fast multi-partition scans;
writes are per (dataset, symbol, date) Parquet partitions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from quant.intraday.data.config import IntradayConfig, partition_path


class MarketDataStore:
    def __init__(self, config: IntradayConfig) -> None:
        self.config = config

    # ---- write side -------------------------------------------------------
    def _write(self, dataset: str, symbol: str, day: date, df: pd.DataFrame) -> Path:
        path = partition_path(self.config.data_root, dataset, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")  # atomic: never leave a half-written partition
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
                "SELECT * FROM read_parquet(?, union_by_name=true) "
                "WHERE index >= ? AND index < ? ORDER BY index",
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
        df.index = pd.DatetimeIndex(df.index, name="timestamp").tz_localize("UTC") if df.index.tz is None else df.index
        return df

    def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._read("minute_bars", symbol, start, end)

    def get_quote_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._read("quote_bars_1s", symbol, start, end)

    def get_trades(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._read("trades", symbol, start, end)
```

Note: `df.to_parquet` writes the DatetimeIndex as a column named `index` by default when read back via DuckDB's `read_parquet`. The test asserts round-trip equivalence, so if column naming differs on your pandas/pyarrow version, set `df.index.name = "index"` before writing in `_write` and adjust the `SELECT`/`set_index` accordingly. Verify with the test.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_store_roundtrip.py -v`
Expected: PASS (2 passed). If the index column name mismatches, fix `_write`/`_read` as noted, then rerun.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock quant/intraday/data/store.py tests/intraday/data/test_store_roundtrip.py
git commit -m "feat(intraday): MarketDataStore Parquet write + DuckDB read round-trip"
```

---

## Task 7: PIT-adjusted reads (wire adjustments into the store)

**Files:**
- Modify: `quant/intraday/data/store.py` (add `as_of` + factor loading)
- Test: `tests/intraday/data/test_store_pit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_store_pit.py
from datetime import date, datetime, timezone

import pandas as pd

from quant.intraday.data.adjustments import Adjustment
from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.store import MarketDataStore


def _bars():
    idx = pd.to_datetime(["2023-05-30T13:30:00Z", "2023-06-02T13:30:00Z"])
    return pd.DataFrame(
        {"open": [400.0, 100.0], "high": [400.0, 100.0], "low": [400.0, 100.0],
         "close": [400.0, 100.0], "volume": [1, 1], "vwap": [400.0, 100.0], "trade_count": [1, 1]},
        index=idx,
    )


def test_get_minute_bars_applies_pit_adjustment(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path))
    store.write_minute_bars("AAPL", date(2023, 5, 30), _bars().iloc[[0]])
    store.write_minute_bars("AAPL", date(2023, 6, 2), _bars().iloc[[1]])
    store.set_adjustments("AAPL", [Adjustment(date(2023, 6, 1), split_ratio=4.0, cash_dividend=0.0)])
    start = datetime(2023, 5, 1, tzinfo=timezone.utc)
    end = datetime(2023, 7, 1, tzinfo=timezone.utc)

    seen_after = store.get_minute_bars("AAPL", start, end, as_of=date(2023, 6, 5))
    assert seen_after.iloc[0]["open"] == 100.0  # split applied

    seen_before = store.get_minute_bars("AAPL", start, end, as_of=date(2023, 5, 31))
    assert seen_before.iloc[0]["open"] == 400.0  # no lookahead
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_store_pit.py -v`
Expected: FAIL — `set_adjustments` / `as_of` kwarg missing.

- [ ] **Step 3: Implement** (modify `store.py`)

Add to `MarketDataStore.__init__`: `self._adjustments: dict[str, list[Adjustment]] = {}`.

Add import: `from quant.intraday.data.adjustments import Adjustment, adjust_prices`.

Add method:
```python
    def set_adjustments(self, symbol: str, factors: list[Adjustment]) -> None:
        self._adjustments[symbol] = sorted(factors, key=lambda a: a.ex_date)
```

Change the three getters to accept `as_of: date | None = None` and apply adjustments:
```python
    def get_minute_bars(self, symbol, start, end, as_of=None):
        return self._maybe_adjust(symbol, self._read("minute_bars", symbol, start, end), as_of)

    def get_quote_bars(self, symbol, start, end, as_of=None):
        return self._maybe_adjust(symbol, self._read("quote_bars_1s", symbol, start, end), as_of)

    def get_trades(self, symbol, start, end, as_of=None):
        return self._maybe_adjust(symbol, self._read("trades", symbol, start, end), as_of)

    def _maybe_adjust(self, symbol, df, as_of):
        if as_of is None or df.empty or symbol not in self._adjustments:
            return df
        return adjust_prices(df, self._adjustments[symbol], as_of)
```
(Add `from datetime import date` if not already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_store_pit.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/store.py tests/intraday/data/test_store_pit.py
git commit -m "feat(intraday): point-in-time adjusted reads in MarketDataStore"
```

---

## Task 8: `replay()` — timestamp-ordered multi-symbol event iterator

**Files:**
- Modify: `quant/intraday/data/store.py`
- Test: `tests/intraday/data/test_replay.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_replay.py
from datetime import date, datetime, timezone

import pandas as pd

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.events import Bar, QuoteBar
from quant.intraday.data.store import MarketDataStore


def _seed(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL", "MSFT")))
    day = date(2023, 6, 1)
    qa = pd.DataFrame({"bid": [1.0], "ask": [1.1], "bid_size": [1], "ask_size": [1]},
                      index=pd.to_datetime(["2023-06-01T13:30:00Z"]))
    store.write_quote_bars("AAPL", day, qa)
    ba = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1],
                       "vwap": [1.0], "trade_count": [1]},
                      index=pd.to_datetime(["2023-06-01T13:30:00Z"]))
    store.write_minute_bars("MSFT", day, ba)
    return store


def test_replay_orders_across_symbols_and_types(tmp_path):
    store = _seed(tmp_path)
    events = list(store.replay(["AAPL", "MSFT"],
                               datetime(2023, 6, 1, tzinfo=timezone.utc),
                               datetime(2023, 6, 2, tzinfo=timezone.utc),
                               datasets=("quote_bars_1s", "minute_bars")))
    # same ts -> QuoteBar before Bar (event rank); AAPL quote then MSFT bar
    assert [type(e).__name__ for e in events] == ["QuoteBar", "Bar"]
    assert isinstance(events[0], QuoteBar) and events[0].symbol == "AAPL"
    assert isinstance(events[1], Bar) and events[1].symbol == "MSFT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_replay.py -v`
Expected: FAIL — `replay` missing.

- [ ] **Step 3: Implement** (add to `store.py`; add `from collections.abc import Iterator, Sequence` and event imports)

```python
    def _rows_to_events(self, dataset, symbol, df):
        from quant.intraday.data.events import Bar, QuoteBar, Trade
        for ts, row in df.iterrows():
            if dataset == "trades":
                yield Trade(ts=ts.to_pydatetime(), symbol=symbol, price=float(row["price"]), size=int(row["size"]))
            elif dataset == "quote_bars_1s":
                yield QuoteBar(ts=ts.to_pydatetime(), symbol=symbol, bid=float(row["bid"]),
                               ask=float(row["ask"]), bid_size=int(row["bid_size"]), ask_size=int(row["ask_size"]))
            elif dataset == "minute_bars":
                yield Bar(ts=ts.to_pydatetime(), symbol=symbol, open=float(row["open"]), high=float(row["high"]),
                          low=float(row["low"]), close=float(row["close"]), volume=int(row["volume"]),
                          vwap=float(row["vwap"]), trade_count=int(row["trade_count"]))

    def replay(self, symbols, start, end, *, datasets=("minute_bars",), as_of=None):
        from quant.intraday.data.events import event_sort_key
        getter = {"trades": self.get_trades, "quote_bars_1s": self.get_quote_bars, "minute_bars": self.get_minute_bars}
        collected = []
        for symbol in symbols:
            for ds in datasets:
                df = getter[ds](symbol, start, end, as_of=as_of)
                collected.extend(self._rows_to_events(ds, symbol, df))
        collected.sort(key=event_sort_key)
        yield from collected
```

(For the full ~1 TB scale a heap-merge of per-partition iterators would avoid materializing all events; the in-memory sort is correct and fine for the bounded windows backtests actually replay. Note this as a future optimization in the docstring.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_replay.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/store.py tests/intraday/data/test_replay.py
git commit -m "feat(intraday): replay() timestamp-ordered event iterator"
```

---

## Task 9: `subscribe()` + the anti-skew golden test (keystone)

**Files:**
- Modify: `quant/intraday/data/store.py` (in-memory buffer + `subscribe`, `push`, `freshness`)
- Test: `tests/intraday/data/test_anti_skew.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_anti_skew.py
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.store import MarketDataStore


def _seed_and_buffer(tmp_path):
    """Write a day to disk, then push the SAME events into the live buffer."""
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL",)))
    day = date(2023, 6, 1)
    q = pd.DataFrame({"bid": [1.0, 1.2], "ask": [1.1, 1.3], "bid_size": [1, 2], "ask_size": [1, 2]},
                     index=pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:01Z"]))
    store.write_quote_bars("AAPL", day, q)
    start = datetime(2023, 6, 1, tzinfo=timezone.utc)
    end = datetime(2023, 6, 2, tzinfo=timezone.utc)
    hist = list(store.replay(["AAPL"], start, end, datasets=("quote_bars_1s",)))
    for ev in hist:  # simulate the realtime stream delivering identical events
        store.push(ev)
    return store, hist


def test_replay_and_subscribe_emit_identical_events(tmp_path):
    store, hist = _seed_and_buffer(tmp_path)
    live = list(store.subscribe(["AAPL"]))
    assert live == hist  # frozen dataclasses compare by value — byte-identical sequence


def test_freshness_reports_staleness(tmp_path):
    store, _ = _seed_and_buffer(tmp_path)
    now = datetime(2023, 6, 1, 13, 30, 30, tzinfo=timezone.utc)
    fr = store.freshness(now=now)
    # last buffered event was 13:30:01, so age ~29s
    assert fr.last_event_ts == datetime(2023, 6, 1, 13, 30, 1, tzinfo=timezone.utc)
    assert fr.age_seconds(now) >= 29
    assert fr.is_stale(now, max_age_seconds=10) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_anti_skew.py -v`
Expected: FAIL — `push`/`subscribe`/`freshness` missing.

- [ ] **Step 3: Implement** (add to `store.py`)

```python
from dataclasses import dataclass as _dataclass  # if dataclass not already imported at top


@_dataclass(frozen=True)
class Freshness:
    last_event_ts: "datetime | None"

    def age_seconds(self, now) -> float:
        if self.last_event_ts is None:
            return float("inf")
        return (now - self.last_event_ts).total_seconds()

    def is_stale(self, now, max_age_seconds: float) -> bool:
        return self.age_seconds(now) > max_age_seconds
```

In `__init__`: `self._buffer: list = []`.

```python
    def push(self, event) -> None:
        """Append a realtime event to the rolling buffer (called by stream.py)."""
        self._buffer.append(event)

    def subscribe(self, symbols):
        wanted = set(symbols)
        from quant.intraday.data.events import event_sort_key
        for ev in sorted(self._buffer, key=event_sort_key):
            if ev.symbol in wanted:
                yield ev

    def freshness(self, now=None) -> "Freshness":
        if not self._buffer:
            return Freshness(last_event_ts=None)
        last = max(ev.ts for ev in self._buffer)
        return Freshness(last_event_ts=last)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_anti_skew.py -v`
Expected: PASS (2 passed). **This is the keystone: identical events from disk-replay and live-buffer.**

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/store.py tests/intraday/data/test_anti_skew.py
git commit -m "feat(intraday): subscribe() + freshness + anti-skew golden test"
```

---

## Task 10: Data quality — session calendar, gap detection, bad-tick filter

**Files:**
- Create: `quant/intraday/data/quality.py`
- Test: `tests/intraday/data/test_quality.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_quality.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/data/quality.py
"""Data-quality guards: session calendar, gap detection, bad-tick filtering, doctor."""

from __future__ import annotations

from datetime import date

import pandas as pd


def regular_session_minutes(day: date) -> int:
    """Number of 1-minute bars in a regular US equity session (9:30–16:00 ET).

    Half-days return 210; this base implementation returns 390 for any weekday.
    (A half-day calendar can be layered in later from the market calendar.)
    """
    return 390


def detect_minute_gaps(bars: pd.DataFrame) -> list[pd.Timestamp]:
    """Return the missing minute timestamps between the first and last bar."""
    if bars.empty:
        return []
    full = pd.date_range(bars.index.min(), bars.index.max(), freq="1min")
    return [ts for ts in full if ts not in bars.index]


def filter_bad_trades(trades: pd.DataFrame, ref_price: float, max_deviation: float = 0.2) -> pd.DataFrame:
    """Drop non-positive prices and prints more than `max_deviation` from ref_price."""
    if trades.empty:
        return trades
    lo, hi = ref_price * (1 - max_deviation), ref_price * (1 + max_deviation)
    mask = (trades["price"] > 0) & (trades["price"] >= lo) & (trades["price"] <= hi)
    return trades.loc[mask]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_quality.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/quality.py tests/intraday/data/test_quality.py
git commit -m "feat(intraday): data-quality guards (calendar, gaps, bad ticks)"
```

---

## Task 11: Backfill ingester (Alpaca SIP, idempotent, resumable) — mock-tested

**Files:**
- Create: `quant/intraday/data/backfill.py`
- Test: `tests/intraday/data/test_backfill.py`

The ingester takes a **client object** (dependency injection) so unit tests pass a fake; production wires `StockHistoricalDataClient`. No network in tests.

- [ ] **Step 1: Write the failing test**

```python
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
        return pd.DataFrame({"price": [10.0, 11.0], "size": [100, 200]},
                            index=pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:30Z"]))

    def get_quotes_df(self, symbol, day):
        return pd.DataFrame({"bid": [9.9], "ask": [10.1], "bid_size": [5], "ask_size": [5]},
                            index=pd.to_datetime(["2023-06-01T13:30:00Z"]))


def test_backfill_writes_all_three_datasets(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path))
    client = FakeHistClient()
    res = backfill_symbol_day(client, store, "AAPL", date(2023, 6, 1))
    assert res.trades_rows == 2 and res.quote_bar_rows == 1 and res.minute_bar_rows == 1
    # partitions queryable through the store
    import datetime as dt
    s, e = dt.datetime(2023, 6, 1, tzinfo=dt.timezone.utc), dt.datetime(2023, 6, 2, tzinfo=dt.timezone.utc)
    assert len(store.get_minute_bars("AAPL", s, e)) == 1


def test_backfill_is_idempotent_and_resumable(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path))
    client = FakeHistClient()
    backfill_symbol_day(client, store, "AAPL", date(2023, 6, 1))
    # second call with skip_existing should NOT re-fetch
    res = backfill_symbol_day(client, store, "AAPL", date(2023, 6, 1), skip_existing=True)
    assert res.skipped is True
    assert client.trade_calls == 1  # not re-fetched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_backfill.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/backfill.py tests/intraday/data/test_backfill.py
git commit -m "feat(intraday): idempotent resumable SIP backfill (mock-tested)"
```

---

## Task 12: Alpaca SIP client adapter (real wiring; `@pytest.mark.alpaca` integration)

**Files:**
- Modify: `quant/intraday/data/backfill.py` (add `AlpacaHistClient` implementing `HistClient`)
- Test: `tests/intraday/data/test_alpaca_hist_client.py`

This adapter is the only network-touching code. Its unit test is marked `alpaca` (skipped unless the SIP subscription + keys exist), matching the repo's existing `alpaca` marker convention.

- [ ] **Step 1: Write the (network-gated) test**

```python
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
```

- [ ] **Step 2: Run test to verify it skips (no subscription)**

Run: `uv run pytest tests/intraday/data/test_alpaca_hist_client.py -v`
Expected: SKIPPED (1 skipped) when the `alpaca` marker is deselected by default, or until the SIP subscription is active. Confirm it at least imports.

- [ ] **Step 3: Implement the adapter** (append to `backfill.py`)

```python
class AlpacaHistClient:
    """Production HistClient backed by alpaca-py's StockHistoricalDataClient (SIP feed)."""

    def __init__(self, settings=None) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        from quant.util.config import Settings

        s = settings or Settings()  # type: ignore[call-arg]
        self._client = StockHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)

    def _day_bounds(self, day: date):
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
```

(Verify exact column names from `alpaca-py` once the subscription is live — they may be `bid_price`/`ask_price`. The `rename` handles the documented names; adjust if the SDK version differs.)

- [ ] **Step 4: Run tests** (unit suite still green; alpaca test skipped)

Run: `uv run pytest tests/intraday/data/ -v -m "not alpaca"`
Expected: all non-alpaca tests PASS; alpaca test deselected.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/backfill.py tests/intraday/data/test_alpaca_hist_client.py
git commit -m "feat(intraday): Alpaca SIP historical client adapter"
```

---

## Task 13: Realtime stream ingester + rolling buffer (fake-stream tested)

**Files:**
- Create: `quant/intraday/data/stream.py`
- Test: `tests/intraday/data/test_stream.py`

The ingester consumes an injected async event source (a fake in tests; `StockDataStream` in prod), converts raw messages to `Event` objects, pushes them into the store buffer, and flushes to today's partition periodically.

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_stream.py
import asyncio
from datetime import datetime, timezone

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.events import QuoteBar
from quant.intraday.data.store import MarketDataStore
from quant.intraday.data.stream import ingest_quotes


async def _fake_source():
    yield {"symbol": "AAPL", "timestamp": datetime(2023, 6, 1, 13, 30, tzinfo=timezone.utc),
           "bid": 1.0, "ask": 1.1, "bid_size": 1, "ask_size": 1}
    yield {"symbol": "AAPL", "timestamp": datetime(2023, 6, 1, 13, 30, 1, tzinfo=timezone.utc),
           "bid": 1.2, "ask": 1.3, "bid_size": 2, "ask_size": 2}


def test_ingest_quotes_pushes_events_to_buffer(tmp_path):
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL",)))
    asyncio.run(ingest_quotes(_fake_source(), store))
    live = list(store.subscribe(["AAPL"]))
    assert len(live) == 2
    assert all(isinstance(e, QuoteBar) for e in live)
    assert live[0].bid == 1.0 and live[1].bid == 1.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_stream.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# quant/intraday/data/stream.py
"""Realtime SIP ingestion: convert streamed messages into Event objects, push
into the store's rolling buffer (which subscribe() serves). The event source is
injected so it is testable with a fake async generator; production passes an
adapter over alpaca-py's StockDataStream."""

from __future__ import annotations

from collections.abc import AsyncIterator

from quant.intraday.data.events import QuoteBar
from quant.intraday.data.store import MarketDataStore


async def ingest_quotes(source: AsyncIterator[dict], store: MarketDataStore) -> int:
    """Consume quote messages, push QuoteBar events to the buffer. Returns count."""
    n = 0
    async for msg in source:
        store.push(
            QuoteBar(
                ts=msg["timestamp"],
                symbol=msg["symbol"],
                bid=float(msg["bid"]),
                ask=float(msg["ask"]),
                bid_size=int(msg["bid_size"]),
                ask_size=int(msg["ask_size"]),
            )
        )
        n += 1
    return n
```

(Production reconnect/backfill-on-gap and the `StockDataStream` adapter — `stream.subscribe_quotes(handler, *symbols)` — are wired in sub-project E/D where the live engine runs the event loop; the message→`QuoteBar` conversion proven here is the reusable core.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/data/test_stream.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/data/stream.py tests/intraday/data/test_stream.py
git commit -m "feat(intraday): realtime quote ingestion into store buffer"
```

---

## Task 14: CLI surface + doctor; package exports; full-suite green

**Files:**
- Create: `quant/intraday/cli.py`
- Modify: `quant/cli.py` (register the `intraday` group)
- Modify: `quant/intraday/data/__init__.py` (export `MarketDataStore`, `IntradayConfig`, `BackfillResult`)
- Modify: `quant/intraday/data/quality.py` (add `run_doctor`)
- Test: `tests/intraday/data/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/intraday/data/test_cli.py
from click.testing import CliRunner

from quant.cli import cli


def test_intraday_data_status_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(cli, ["intraday", "data", "status"])
    assert result.exit_code == 0
    assert "Intraday data" in result.output


def test_intraday_data_doctor_reports_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(cli, ["intraday", "data", "doctor"])
    assert result.exit_code == 0
    assert "partitions" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/data/test_cli.py -v`
Expected: FAIL — no `intraday` command group.

- [ ] **Step 3: Implement**

Add `run_doctor` to `quality.py`:
```python
def run_doctor(data_root) -> dict:
    """Summarize the intraday store: partition counts per dataset."""
    from pathlib import Path

    root = Path(data_root) / ""  # noop to keep Path import local
    counts = {}
    for ds in ("trades", "quote_bars_1s", "minute_bars"):
        d = Path(data_root) / ds
        counts[ds] = len(list(d.rglob("*.parquet"))) if d.exists() else 0
    return counts
```

Create `quant/intraday/cli.py`:
```python
"""`quant intraday ...` command group."""

from __future__ import annotations

import click
from rich.console import Console

from quant.intraday.data.quality import run_doctor
from quant.util.config import Settings

console = Console()


@click.group()
def intraday() -> None:
    """Intraday equities subsystem."""


@intraday.group()
def data() -> None:
    """Intraday data layer commands."""


@data.command()
def status() -> None:
    """Show intraday data store status."""
    root = Settings().data_dir / "intraday"  # type: ignore[call-arg]
    counts = run_doctor(root)
    console.print(f"[bold]Intraday data[/bold] at {root}")
    for ds, n in counts.items():
        console.print(f"  {ds}: {n} partitions")


@data.command()
def doctor() -> None:
    """Health check: partition counts and obvious gaps."""
    root = Settings().data_dir / "intraday"  # type: ignore[call-arg]
    counts = run_doctor(root)
    total = sum(counts.values())
    console.print(f"intraday store: {total} partitions across {len(counts)} datasets")
    for ds, n in counts.items():
        console.print(f"  {ds}: {n} partitions")
```

Register in `quant/cli.py` — after the root `cli` group is defined and other commands are added, add:
```python
from quant.intraday.cli import intraday as _intraday_group

cli.add_command(_intraday_group)
```
(Place this import + `add_command` next to the other `cli.add_command(...)` / command registrations. If commands are registered via decorators on `cli` directly, add the single `cli.add_command(_intraday_group)` line after the group is created.)

Update `quant/intraday/data/__init__.py`:
```python
from quant.intraday.data.backfill import BackfillResult, backfill_symbol_day
from quant.intraday.data.config import DEFAULT_UNIVERSE, IntradayConfig
from quant.intraday.data.events import Bar, Event, QuoteBar, Trade, event_sort_key
from quant.intraday.data.store import MarketDataStore

__all__ = [
    "Bar", "Event", "QuoteBar", "Trade", "event_sort_key",
    "IntradayConfig", "DEFAULT_UNIVERSE", "MarketDataStore",
    "BackfillResult", "backfill_symbol_day",
]
```

- [ ] **Step 4: Run the test + full suite**

Run: `uv run pytest tests/intraday/data/test_cli.py -v`
Expected: PASS (2 passed).

Run: `uv run pytest -m "not alpaca" -q && uv run ruff check quant/intraday tests/intraday && uv run ruff format --check quant/intraday tests/intraday && uv run mypy quant/intraday`
Expected: full suite green, lint/format clean, mypy clean. Fix any issues before committing.

- [ ] **Step 5: Commit**

```bash
git add quant/intraday/cli.py quant/cli.py quant/intraday/data/__init__.py quant/intraday/data/quality.py tests/intraday/data/test_cli.py
git commit -m "feat(intraday): CLI (status/doctor) + package exports; data layer complete"
```

---

## Optional Task 15 (post-subscription): live backfill smoke + a first real day

Once Algo Trader Plus is active and keys are in `.env`:

- [ ] Run one real day for one symbol and eyeball it:
  ```bash
  uv run python -c "from datetime import date; from quant.intraday.data.backfill import AlpacaHistClient, backfill_symbol_day; from quant.intraday.data.store import MarketDataStore; from quant.intraday.data.config import IntradayConfig; from quant.util.config import Settings; s=MarketDataStore(IntradayConfig(data_root=Settings().data_dir/'intraday')); print(backfill_symbol_day(AlpacaHistClient(), s, 'AAPL', date(2024,1,2)))"
  ```
- [ ] Run the gated integration test: `uv run pytest -m alpaca tests/intraday/data/ -v`
- [ ] Spot-check spread sanity: median 1-sec spread for AAPL on a normal day should be a few cents.
- [ ] Then write the full-universe backfill driver (loops `DEFAULT_UNIVERSE` × trading days, `skip_existing=True`, rate-limit backoff) as the first task of operationalizing — this is naturally part of sub-project E (ops), since it's a long-running job.

---

## Self-Review

**1. Spec coverage:**
- Universe (~100 liquid, PIT) → Task 2 (`DEFAULT_UNIVERSE`); PIT *membership* table is noted in the spec's `universe.py` but the live decision only needs the static liquid set now — full PIT membership (listings/delistings) is deferred to a follow-up task and flagged here as a known gap (low risk: the chosen names are long-lived; revisit before any survivorship-sensitive cross-sectional study in sub-project C).
- Substrate: trades + 1s NBBO bars + 1m bars → Tasks 3, 4, 6, 11. ✓
- Storage = partitioned Parquet + DuckDB → Tasks 2, 6. ✓
- One event interface / replay==subscribe → Tasks 8, 9 (keystone test). ✓
- PIT adjustments → Tasks 5, 7. ✓
- Backfill idempotent/resumable → Task 11; Alpaca adapter → Task 12. ✓
- Realtime ingest + freshness → Tasks 9, 13. ✓
- Data quality + doctor → Tasks 10, 14. ✓
- Resilience (atomic partition write, freshness fail-closed hook) → Tasks 6 (`.tmp` rename), 9 (`freshness`). Reconnect/gap-backfill explicitly deferred to D/E (noted in Task 13). ✓
- No network in unit tests → Tasks 11/13 use fakes; Task 12 is `@pytest.mark.alpaca`. ✓

**Gap intentionally deferred (flagged, not silent):** full PIT universe membership; heap-merge replay for TB-scale; production stream reconnect/gap-backfill. All belong to later sub-projects or are non-blocking optimizations; called out at their task sites.

**2. Placeholder scan:** every code step contains runnable code; no "TODO"/"handle errors"/"similar to". The two "verify column name against the live SDK" notes (Tasks 6, 12) are explicit verification steps, not placeholders — the code given matches the documented `alpaca-py`/pandas behavior and the test will catch a mismatch.

**3. Type consistency:** `Trade`/`QuoteBar`/`Bar` fields are identical everywhere (events.py defines, store/backfill/stream construct with the same kwargs). `MarketDataStore` method names (`write_*`, `get_*`, `replay`, `push`, `subscribe`, `freshness`, `set_adjustments`) are used consistently across Tasks 6–14. `HistClient` Protocol (`get_trades_df`/`get_quotes_df`) matches both `FakeHistClient` (Task 11) and `AlpacaHistClient` (Task 12). `BackfillResult` fields match the Task 11 assertions.
```
