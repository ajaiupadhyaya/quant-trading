"""Order-template dataclass + client_order_id helper for per-strategy attribution."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from datetime import date


class OrderSide(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(enum.StrEnum):
    DAY = "day"
    GTC = "gtc"


@dataclass(frozen=True)
class OrderTemplate:
    """A target order to be submitted to Alpaca.

    `qty` is always a positive integer. `side` encodes direction. The execution
    fields default to MARKET / DAY / no-limit-price, reproducing the historical
    behavior byte-for-byte; non-default values are foundation for future
    execution-quality work and are not emitted by any live path today.
    """

    symbol: str
    qty: int
    side: OrderSide
    strategy_slug: str
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    def __post_init__(self) -> None:
        if self.order_type is OrderType.LIMIT:
            if (
                self.limit_price is None
                or not math.isfinite(self.limit_price)
                or self.limit_price <= 0
            ):
                raise ValueError("LIMIT order requires a positive, finite limit_price")
        elif self.limit_price is not None:
            raise ValueError("MARKET order must not carry a limit_price")


def make_client_order_id(strategy_slug: str, symbol: str, dt: date) -> str:
    """Deterministic id: ``<slug>-<YYYYMMDD>-<symbol>``.

    Deterministic per (strategy, symbol, session-date) so a resubmission of the
    same logical order on the same day collides on client_order_id and Alpaca
    rejects the duplicate — broker-level idempotency that guards against a
    crash-then-retry double-submit. The slug prefix still attributes fills to a
    strategy. (Alpaca caps client_order_id at 48 chars; slugs+symbols here are
    well under that.)
    """
    return f"{strategy_slug}-{dt:%Y%m%d}-{symbol}"
