# quant/intraday/data/stream.py
"""Realtime SIP ingestion: convert streamed messages into Event objects, push
into the store's rolling buffer (which subscribe() serves). The event source is
injected so it is testable with a fake async generator; production passes an
adapter over alpaca-py's StockDataStream."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from quant.intraday.data.events import QuoteBar
from quant.intraday.data.store import MarketDataStore


async def ingest_quotes(source: AsyncIterator[dict[str, Any]], store: MarketDataStore) -> int:
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
