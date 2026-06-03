"""Phase 7B engine integration: negative-news detector + loop news throttle."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant.engine.events import EventConfig, detect_events
from quant.engine.loop import EngineConfig, engine_dir, run_engine
from quant.nlp.sentiment import NewsSentiment
from tests.engine.conftest import fake_settings, mk_state

CFG = EventConfig()


def _ns(mean: float, n: int = 8, top: str | None = "Bank plunges on bankruptcy") -> NewsSentiment:
    return NewsSentiment(mean, n, 1, max(0, n - 1), 0.6, top, -1.0, "t")


def test_negative_news_warn_and_critical() -> None:
    warn = detect_events(mk_state(), mk_state(news_sentiment=-0.40, news_n_items=8), CFG)
    e = next(e for e in warn if e.code == "negative_news")
    assert e.severity == "warn"
    crit = detect_events(mk_state(), mk_state(news_sentiment=-0.55, news_n_items=8), CFG)
    assert next(e for e in crit if e.code == "negative_news").severity == "critical"


def test_negative_news_needs_min_items() -> None:
    evs = detect_events(mk_state(), mk_state(news_sentiment=-0.8, news_n_items=2), CFG)
    assert not any(e.code == "negative_news" for e in evs)  # too few items to trust


def test_positive_news_does_not_fire() -> None:
    evs = detect_events(mk_state(), mk_state(news_sentiment=0.2, news_n_items=10), CFG)
    assert not any(e.code == "negative_news" for e in evs)


def _clock(start: datetime, step: float):
    t = {"v": start - timedelta(seconds=step)}

    def now() -> datetime:
        t["v"] += timedelta(seconds=step)
        return t["v"]

    return now


def _run(tmp_path: Path, news_fn, *, refresh_s: float, cycles: int = 3):
    return run_engine(
        fake_settings(tmp_path),
        max_cycles=cycles,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 46),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        intraday_fn=lambda: None,
        news_fn=news_fn,
        config=EngineConfig(news_refresh_s=refresh_s),
    )


def test_loop_news_throttled(tmp_path: Path) -> None:
    calls = {"n": 0}

    def news_fn():
        calls["n"] += 1
        return _ns(-0.1)

    _run(tmp_path, news_fn, refresh_s=180.0)  # 46s steps << 180s -> fetched once, reused
    assert calls["n"] == 1


def test_loop_news_refetches_when_stale(tmp_path: Path) -> None:
    calls = {"n": 0}

    def news_fn():
        calls["n"] += 1
        return _ns(-0.1)

    _run(tmp_path, news_fn, refresh_s=0.0)  # always stale -> refetch each cycle
    assert calls["n"] == 3


def test_loop_news_flows_into_state(tmp_path: Path) -> None:
    _run(tmp_path, lambda: _ns(-0.35, n=9), refresh_s=180.0)
    state = json.loads((engine_dir(tmp_path) / "state.json").read_text())
    assert abs(state["news_sentiment"] - (-0.35)) < 1e-9
    assert state["news_n_items"] == 9
