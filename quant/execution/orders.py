"""Order-template dataclass + client_order_id helper for per-strategy attribution."""

from __future__ import annotations

import enum
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
    """Deterministic id: ``<slug>-<YYYYMMDD>-<symbol>``.

    Deterministic per (strategy, symbol, session-date) so a resubmission of the
    same logical order on the same day collides on client_order_id and Alpaca
    rejects the duplicate — broker-level idempotency that guards against a
    crash-then-retry double-submit. The slug prefix still attributes fills to a
    strategy. (Alpaca caps client_order_id at 48 chars; slugs+symbols here are
    well under that.)
    """
    return f"{strategy_slug}-{dt:%Y%m%d}-{symbol}"
