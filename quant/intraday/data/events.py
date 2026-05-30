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
