"""Order-template dataclass + client_order_id helper for per-strategy attribution."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from datetime import date


class OrderSide(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class OrderTemplate:
    """A target order to be submitted to Alpaca.

    `qty` is always a positive integer. `side` encodes direction.
    """

    symbol: str
    qty: int
    side: OrderSide
    strategy_slug: str


def make_client_order_id(strategy_slug: str, symbol: str, dt: date) -> str:
    """Format: <slug>-<YYYYMMDD>-<symbol>-<uuid8>.

    The slug prefix is how we attribute fills back to a specific strategy when
    multiple strategies share a single Alpaca account.
    """
    return f"{strategy_slug}-{dt:%Y%m%d}-{symbol}-{uuid.uuid4().hex[:8]}"
