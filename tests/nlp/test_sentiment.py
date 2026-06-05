"""Local financial-sentiment scorer + aggregation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant.data.news import NewsItem
from quant.nlp import sentiment as snt
from quant.nlp.lexicon import NEGATIVE, POSITIVE
from quant.nlp.sentiment import (
    NewsSentiment,
    live_news_sentiment,
    render_sentiment,
    score_news,
    score_text,
)


def test_lexicon_is_disjoint() -> None:
    assert frozenset() == POSITIVE & NEGATIVE  # no word may be both pos and neg


def test_score_text_polarity() -> None:
    assert score_text("Apple beats earnings, stock surges to record high") == 1.0
    assert score_text("Bank plunges on fraud probe and bankruptcy fears") == -1.0
    assert score_text("Fed holds rates steady amid mixed signals") == 0.0


def test_negation_flips_polarity() -> None:
    # "not miss" -> positive; "profit"/"grows" positive -> net positive
    assert score_text("Company does not miss expectations, profit grows") > 0
    # a lone negated positive flips negative
    assert score_text("revenue did not grow") < 0


def test_score_in_unit_range() -> None:
    for txt in ["surge rally beat", "plunge crash loss", "the company reported"]:
        assert -1.0 <= score_text(txt) <= 1.0


def _item(headline: str, summary: str = "", created: str = "2026-06-03T12:00:00+00:00") -> NewsItem:
    return NewsItem(
        id=headline[:8],
        created_at=created,
        headline=headline,
        summary=summary,
        source="benzinga",
        symbols=("SPY",),
        url=None,
    )


def test_score_news_aggregates() -> None:
    items = [
        _item("Stock surges to record profit beat"),
        _item("Shares plunge on bankruptcy and fraud probe"),
        _item("Market plummets amid recession fears and lawsuit"),
        _item("Company holds annual meeting"),
    ]
    s = score_news(items)
    assert s.n_items == 4
    assert s.mean_sentiment is not None and s.mean_sentiment < 0  # 2 strong negatives
    assert s.n_negative >= 2 and s.n_positive >= 1
    assert s.negative_frac is not None and s.negative_frac >= 0.5
    assert s.top_negative_headline is not None  # the worst headline is surfaced


def test_score_news_empty() -> None:
    e = score_news([])
    assert e.n_items == 0 and e.mean_sentiment is None


def test_render_sentiment() -> None:
    assert "no recent items" in render_sentiment(score_news([]))
    s = NewsSentiment(-0.3, 5, 1, 3, 0.6, "Bank collapses", -1.0, "t")
    out = render_sentiment(s)
    assert "-0.30" in out and "Bank collapses" in out


def test_live_news_sentiment_failopen(monkeypatch: pytest.MonkeyPatch) -> None:
    # fetch raising must degrade to an empty read, never propagate.
    monkeypatch.setattr(snt, "fetch_news", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    s = live_news_sentiment(SimpleNamespace())
    assert s.n_items == 0 and s.mean_sentiment is None
