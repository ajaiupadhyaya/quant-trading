"""Unique client_order_id generation for the intraday sleeve. Namespaced 'sleeve:'
so attribution and reconciliation can isolate sleeve orders from daily-system ones."""

from __future__ import annotations

from datetime import datetime


def make_sleeve_coid(symbol: str, ts: datetime, seq: int) -> str:
    """sleeve:<SYMBOL>:<epoch-seconds>:<seq> — unique per (symbol, tick, order index)."""
    return f"sleeve:{symbol}:{int(ts.timestamp())}:{seq}"
