# tests/intraday/data/test_events.py
from datetime import datetime, timezone

from quant.intraday.data.events import Bar, QuoteBar, Trade, event_sort_key


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_quotebar_mid_and_spread():
    q = QuoteBar(ts=_ts("2023-06-01T13:30:00"), symbol="AAPL", bid=100.0, ask=100.04, bid_size=5, ask_size=7)
    assert round(q.mid, 4) == 100.02
    assert round(q.spread, 4) == 0.04


def test_event_sort_key_orders_by_ts_then_type_then_symbol():
    t = _ts("2023-06-01T13:30:00")
    quote = QuoteBar(ts=t, symbol="MSFT", bid=1, ask=2, bid_size=1, ask_size=1)
    trade = Trade(ts=t, symbol="AAPL", price=1.5, size=10)
    bar = Bar(ts=t, symbol="AAPL", open=1, high=2, low=1, close=2, volume=10, vwap=1.5, trade_count=3)
    events = [bar, trade, quote]
    events.sort(key=event_sort_key)
    # same ts -> QuoteBar(0) before Trade(1) before Bar(2)
    assert [type(e).__name__ for e in events] == ["QuoteBar", "Trade", "Bar"]


def test_events_are_frozen():
    import dataclasses
    import pytest

    t = Trade(ts=_ts("2023-06-01T13:30:00"), symbol="AAPL", price=1.0, size=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.price = 2.0  # type: ignore[misc]
