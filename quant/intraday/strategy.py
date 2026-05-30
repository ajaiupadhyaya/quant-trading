"""Event-driven intraday strategy interface. SEPARATE from the daily pull-based
quant.strategies.base.Strategy — intraday strategies react to an event stream.
The same on_event() drives the backtester (B) and the live engine (D), which is
the execution-side guarantee against train/serve skew."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from quant.intraday.data.events import Event, QuoteBar


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    qty: int
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Order qty must be positive, got {self.qty}")
        if self.type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("MARKET order must not set limit_price")


@runtime_checkable
class StrategyContext(Protocol):
    """Read-only market+account view exposing ONLY data with ts <= the current event."""

    def position(self, symbol: str) -> int: ...
    def cash(self) -> float: ...
    def nbbo(self, symbol: str) -> QuoteBar | None: ...
    def now(self) -> datetime: ...


@runtime_checkable
class IntradayStrategy(Protocol):
    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]: ...
