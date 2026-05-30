# tests/intraday/data/test_anti_skew.py
from datetime import UTC, date, datetime

import pandas as pd

from quant.intraday.data.config import IntradayConfig
from quant.intraday.data.events import QuoteBar
from quant.intraday.data.store import MarketDataStore


def _seed_and_buffer(tmp_path):
    """Write a day to disk, then push the SAME events into the live buffer."""
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL",)))
    day = date(2023, 6, 1)
    q = pd.DataFrame(
        {"bid": [1.0, 1.2], "ask": [1.1, 1.3], "bid_size": [1, 2], "ask_size": [1, 2]},
        index=pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:01Z"]),
    )
    store.write_quote_bars("AAPL", day, q)
    start = datetime(2023, 6, 1, tzinfo=UTC)
    end = datetime(2023, 6, 2, tzinfo=UTC)
    hist = list(store.replay(["AAPL"], start, end, datasets=("quote_bars_1s",)))
    for ev in hist:  # simulate the realtime stream delivering identical events
        store.push(ev)
    return store, hist


def test_replay_and_subscribe_emit_identical_events(tmp_path):
    store, hist = _seed_and_buffer(tmp_path)
    live = list(store.subscribe(["AAPL"]))
    assert live == hist  # frozen dataclasses compare by value — byte-identical sequence


def test_replay_and_subscribe_identical_when_built_independently(tmp_path):
    """Non-circular keystone: events read from disk and events constructed
    independently (as the live stream would build them from raw messages) must
    produce identical sequences — even when pushed in a different order."""
    store = MarketDataStore(IntradayConfig(data_root=tmp_path, universe=("AAPL",)))
    day = date(2023, 6, 1)
    q = pd.DataFrame(
        {"bid": [1.0, 1.2], "ask": [1.1, 1.3], "bid_size": [1, 2], "ask_size": [1, 2]},
        index=pd.to_datetime(["2023-06-01T13:30:00Z", "2023-06-01T13:30:01Z"]),
    )
    store.write_quote_bars("AAPL", day, q)
    # Built independently from raw values, pushed in REVERSE order on purpose.
    live_events = [
        QuoteBar(
            ts=datetime(2023, 6, 1, 13, 30, 1, tzinfo=UTC),
            symbol="AAPL",
            bid=1.2,
            ask=1.3,
            bid_size=2,
            ask_size=2,
        ),
        QuoteBar(
            ts=datetime(2023, 6, 1, 13, 30, 0, tzinfo=UTC),
            symbol="AAPL",
            bid=1.0,
            ask=1.1,
            bid_size=1,
            ask_size=1,
        ),
    ]
    for ev in live_events:
        store.push(ev)
    s = datetime(2023, 6, 1, tzinfo=UTC)
    e = datetime(2023, 6, 2, tzinfo=UTC)
    assert list(store.subscribe(["AAPL"])) == list(
        store.replay(["AAPL"], s, e, datasets=("quote_bars_1s",))
    )


def test_freshness_reports_staleness(tmp_path):
    store, _ = _seed_and_buffer(tmp_path)
    now = datetime(2023, 6, 1, 13, 30, 30, tzinfo=UTC)
    fr = store.freshness(now=now)
    # last buffered event was 13:30:01, so age ~29s
    assert fr.last_event_ts == datetime(2023, 6, 1, 13, 30, 1, tzinfo=UTC)
    assert fr.age_seconds(now) >= 29
    assert fr.is_stale(now, max_age_seconds=10) is True
