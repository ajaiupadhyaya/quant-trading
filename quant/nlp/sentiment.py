"""Local financial-sentiment scoring for market news (no LLM, no heavy deps).

A deterministic lexicon scorer with light negation handling turns a headline
into a polarity in [-1, +1]; ``score_news`` aggregates a batch into a
``NewsSentiment`` read (mean polarity, negative fraction, most-negative
headline). ``live_news_sentiment`` is the fail-open fetch+score entrypoint the
engine loop and the Claude analyst both use. Pluggable by design: swap the
``score_text`` backend for a model (FinBERT) later without touching callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from quant.data.news import NewsItem, fetch_news
from quant.nlp.lexicon import NEGATIVE, NEGATORS, POSITIVE
from quant.util.logging import logger

_TOKEN = re.compile(r"[a-z][a-z'\-]+")
_NEG_WINDOW = 3  # a negator this many tokens back flips the polarity
_NEUTRAL_BAND = 0.05  # |score| <= this counts as neutral for the pos/neg tallies


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def score_text(text: str) -> float:
    """Polarity of one piece of text in [-1, +1]; 0.0 when no sentiment words."""
    toks = tokenize(text)
    pos = neg = 0
    for i, t in enumerate(toks):
        if t in POSITIVE:
            polarity = 1
        elif t in NEGATIVE:
            polarity = -1
        else:
            continue
        if any(toks[j] in NEGATORS for j in range(max(0, i - _NEG_WINDOW), i)):
            polarity = -polarity
        if polarity > 0:
            pos += 1
        else:
            neg += 1
    total = pos + neg
    return (pos - neg) / total if total else 0.0


@dataclass(frozen=True)
class NewsSentiment:
    mean_sentiment: float | None  # average headline polarity, [-1, +1]
    n_items: int
    n_positive: int
    n_negative: int
    negative_frac: float | None
    top_negative_headline: str | None
    top_negative_score: float | None
    asof: str | None  # most-recent item timestamp (freshness)


_EMPTY = NewsSentiment(None, 0, 0, 0, None, None, None, None)


def score_news(items: list[NewsItem]) -> NewsSentiment:
    """Aggregate a batch of headlines into one sentiment read. Never raises."""
    if not items:
        return _EMPTY
    scored: list[tuple[NewsItem, float]] = []
    for it in items:
        text = it.headline + (". " + it.summary if it.summary else "")
        scored.append((it, score_text(text)))

    vals = [s for _, s in scored]
    n = len(vals)
    mean = sum(vals) / n
    n_pos = sum(1 for v in vals if v > _NEUTRAL_BAND)
    n_neg = sum(1 for v in vals if v < -_NEUTRAL_BAND)
    worst_item, worst_score = min(scored, key=lambda x: x[1])
    asof = max((it.created_at for it in items), default=None)

    return NewsSentiment(
        mean_sentiment=mean,
        n_items=n,
        n_positive=n_pos,
        n_negative=n_neg,
        negative_frac=(n_neg / n if n else None),
        top_negative_headline=(worst_item.headline if worst_score < -_NEUTRAL_BAND else None),
        top_negative_score=(worst_score if worst_score < -_NEUTRAL_BAND else None),
        asof=asof,
    )


def live_news_sentiment(
    settings: Any, symbols: list[str] | None = None, *, lookback_minutes: int = 240
) -> NewsSentiment:
    """Fetch recent news + score it, fail-open to an empty read (the loop/analyst seam)."""
    try:
        items = fetch_news(settings, symbols, lookback_minutes=lookback_minutes)
        return score_news(items)
    except Exception as exc:  # fail-open
        logger.info("nlp.sentiment: live news sentiment skipped ({!r})", exc)
        return _EMPTY


def render_sentiment(s: NewsSentiment | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if s is None or s.n_items == 0 or s.mean_sentiment is None:
        return "News sentiment: no recent items"
    head = (
        f"News sentiment: {s.mean_sentiment:+.2f} over {s.n_items} items "
        f"({s.n_negative} neg / {s.n_positive} pos)"
    )
    if s.top_negative_headline:
        head += f' | worst: "{s.top_negative_headline[:90]}"'
    return head
