"""Market news ingestion (Alpaca / Benzinga) — bounded, fail-open.

Pulls recent headlines for a symbol set into plain ``NewsItem`` records that the
local sentiment scorer (``quant.nlp.sentiment``) consumes. No persistence, no
LLM; one bounded network call that degrades to ``[]`` on any failure.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from quant.data.universe import ETF_UNIVERSE
from quant.util.logging import logger


@dataclass(frozen=True)
class NewsItem:
    id: str
    created_at: str  # ISO timestamp
    headline: str
    summary: str
    source: str | None
    symbols: tuple[str, ...]
    url: str | None


def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def _to_item(raw: Any) -> NewsItem | None:
    try:
        headline = getattr(raw, "headline", None)
        if not headline:
            return None
        created = getattr(raw, "created_at", None)
        created_iso = ""
        if created is not None:
            iso = getattr(created, "isoformat", None)
            created_iso = iso() if callable(iso) else str(created)
        syms = getattr(raw, "symbols", None) or ()
        return NewsItem(
            id=str(getattr(raw, "id", "") or ""),
            created_at=created_iso,
            headline=str(headline),
            summary=str(getattr(raw, "summary", "") or ""),
            source=(str(getattr(raw, "source", "")) or None),
            symbols=tuple(str(s) for s in syms),
            url=(str(getattr(raw, "url", "")) or None),
        )
    except Exception:
        return None


def fetch_news(
    settings: Any,
    symbols: list[str] | None = None,
    *,
    lookback_minutes: int = 240,
    limit: int = 50,
    timeout_s: float = 10.0,
) -> list[NewsItem]:
    """Recent headlines for ``symbols`` (default ETF universe). ``[]`` on failure."""
    syms = symbols if symbols is not None else list(ETF_UNIVERSE)
    start = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key)
        req = NewsRequest(symbols=",".join(syms), start=start, limit=limit)
        res = _with_timeout(lambda: client.get_news(req), timeout_s)
    except Exception as exc:  # fail-open
        logger.info("data.news: fetch skipped ({!r})", exc)
        return []

    raw_items: Any = []
    data = getattr(res, "data", None)
    if isinstance(data, dict):
        raw_items = data.get("news", [])
    if not raw_items:
        raw_items = getattr(res, "news", []) or []

    items = [it for it in (_to_item(r) for r in raw_items) if it is not None]
    # De-dup by id (Alpaca can repeat across overlapping symbol tags).
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for it in items:
        key = it.id or it.headline
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return unique
