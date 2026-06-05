"""Alpaca news fetcher: parsing, dedup, fail-open."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant.data import news as nm
from quant.data.news import NewsItem, fetch_news


def _raw(nid: str, headline: str, syms=("SPY",)):
    return SimpleNamespace(
        id=nid,
        created_at="2026-06-03T12:00:00+00:00",
        headline=headline,
        summary="body",
        source="benzinga",
        symbols=list(syms),
        url="http://x",
    )


def test_to_item_parses_and_skips_headless() -> None:
    it = nm._to_item(_raw("1", "Headline here"))
    assert isinstance(it, NewsItem) and it.headline == "Headline here"
    assert nm._to_item(SimpleNamespace(headline=None)) is None  # no headline -> dropped


def test_fetch_parses_and_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = SimpleNamespace(
        data={"news": [_raw("1", "A surges"), _raw("1", "A surges"), _raw("2", "B falls")]}
    )
    monkeypatch.setattr(nm, "_with_timeout", lambda fn, seconds: resp)
    settings = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key="s")
    items = fetch_news(settings, ["SPY"])
    assert [i.id for i in items] == ["1", "2"]  # duplicate id collapsed


def test_fetch_failopen_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nm, "_with_timeout", lambda fn, seconds: (_ for _ in ()).throw(RuntimeError())
    )
    settings = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key="s")
    assert fetch_news(settings, ["SPY"]) == []


def test_fetch_handles_news_attribute_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some SDK versions expose .news instead of .data["news"].
    resp = SimpleNamespace(data=None, news=[_raw("9", "C rallies")])
    monkeypatch.setattr(nm, "_with_timeout", lambda fn, seconds: resp)
    settings = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key="s")
    items = fetch_news(settings, ["SPY"])
    assert len(items) == 1 and items[0].headline == "C rallies"
