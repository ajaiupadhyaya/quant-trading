"""Live intraday quote feed: REST-polls Alpaca latest NBBO quotes (sufficient at a
60s cadence) and emits QuoteBar events. The broker data client is injected so the
loop and tests can substitute a fake; real construction is via from_settings()."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from quant.intraday.data.events import QuoteBar


class FeedError(RuntimeError):
    """Raised when the upstream quote source fails. The loop must NOT trade on stale
    data — it skips new actions for the tick and retries next tick."""


class _DataClient(Protocol):
    def get_stock_latest_quote(self, request: Any) -> dict[str, Any]: ...


class LiveQuoteFeed:
    def __init__(self, *, symbols: list[str], data_client: _DataClient) -> None:
        self._symbols = symbols
        self._client = data_client

    @classmethod
    def from_settings(cls, *, symbols: list[str], settings: Any = None) -> LiveQuoteFeed:
        from alpaca.data.historical import StockHistoricalDataClient

        from quant.util.config import Settings

        s = settings or Settings()  # type: ignore[call-arg]
        client = StockHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)
        return cls(symbols=symbols, data_client=client)

    def latest_quotes(self, now: datetime | None = None) -> list[QuoteBar]:
        from alpaca.data.requests import StockLatestQuoteRequest

        ts = now or datetime.now(UTC)
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=self._symbols)
            raw = self._client.get_stock_latest_quote(req)
        except Exception as exc:  # normalize any upstream error to FeedError
            raise FeedError(str(exc)) from exc
        bars: list[QuoteBar] = []
        for sym, q in raw.items():
            bars.append(
                QuoteBar(
                    ts=ts,
                    symbol=sym,
                    bid=float(q.bid_price),
                    ask=float(q.ask_price),
                    bid_size=int(q.bid_size),
                    ask_size=int(q.ask_size),
                )
            )
        return bars
