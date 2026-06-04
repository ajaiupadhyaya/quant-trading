"""The continuous loop: persistence, event/Slack/Claude handling, fail-safety."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant.engine import loop as lp
from quant.engine.loop import EngineConfig, engine_dir, run_engine
from tests.engine.conftest import SpySlack, fake_settings, mk_state


@pytest.fixture(autouse=True)
def _hermetic_fundamentals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the loop tests offline: the default fundamentals_fn would otherwise
    issue ~20 cold SEC EDGAR fetches per refresh. These tests stub
    build_market_state and never assert on the fundamentals read."""
    monkeypatch.setattr(lp, "live_fundamentals", lambda *_a, **_k: None)


def _clock(start: datetime, step_s: float):
    t = {"v": start - timedelta(seconds=step_s)}

    def now() -> datetime:
        t["v"] += timedelta(seconds=step_s)
        return t["v"]

    return now


def _script(monkeypatch, states):
    """Make build_market_state return scripted states in order (last repeats)."""
    i = {"n": 0}

    def fake_build(*a: object, **k: object):
        s = states[min(i["n"], len(states) - 1)]
        i["n"] += 1
        return s

    monkeypatch.setattr(lp, "build_market_state", fake_build)


def test_persists_state_jsonl_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, [mk_state(), mk_state()])
    s = fake_settings(tmp_path)
    spy = SpySlack()
    run_engine(
        s,
        max_cycles=2,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 46),
        positions_fn=lambda: {"GLD": 1},
        equity_fn=lambda: 1_000_000.0,
        slack=spy,
    )
    edir = engine_dir(tmp_path)
    assert (edir / "state.json").exists()
    assert len([ln for ln in (edir / "state.jsonl").read_text().splitlines() if ln.strip()]) == 2
    hb = json.loads((edir / "heartbeat.json").read_text())
    assert hb["cycle"] == 1
    assert spy.msgs == []  # dry_run posts nothing


def test_events_posted_and_critical_escalated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cycle1 calm, cycle2 risk-off + risk breach -> posture_cross + risk_breach (both critical)
    _script(monkeypatch, [mk_state(), mk_state(composite_label="risk-off", port_var_95=0.06)])
    s = fake_settings(tmp_path)
    spy = SpySlack()
    claude_calls: list[int] = []

    def claude_fn(state, events, settings) -> str:
        claude_calls.append(len(events))
        return "ANALYSIS"

    run_engine(
        s,
        max_cycles=2,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 46),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        slack=spy,
        claude_fn=claude_fn,
    )
    assert any("risk_breach" in m for m in spy.msgs)
    assert any("posture_cross" in m for m in spy.msgs)
    assert len(claude_calls) == 1  # one escalation for the fresh critical events
    assert any("engine analysis" in m for m in spy.msgs)


def test_claude_min_gap_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crit = mk_state(port_var_95=0.06, composite_label="risk-off")
    # three consecutive critical cycles, but dedup window 0 so each is 'fresh'
    _script(monkeypatch, [crit, mk_state(), crit])
    s = fake_settings(tmp_path)
    calls: list[int] = []
    run_engine(
        s,
        max_cycles=3,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 60),  # 60s steps << 1800s gap
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        slack=SpySlack(),
        claude_fn=lambda a, b, c: calls.append(1) or "x",
        config=EngineConfig(event_dedup_window_s=0.0),
    )
    assert len(calls) == 1  # min-gap (1800s) blocks the later escalations


def test_claude_session_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crit = mk_state(port_var_95=0.06)
    _script(monkeypatch, [crit, mk_state(), crit])
    s = fake_settings(tmp_path)
    calls: list[int] = []
    run_engine(
        s,
        max_cycles=3,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 60),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        slack=SpySlack(),
        claude_fn=lambda a, b, c: calls.append(1) or "x",
        config=EngineConfig(claude_min_gap_s=0.0, claude_max_per_session=1, event_dedup_window_s=0.0),
    )
    assert len(calls) == 1  # capped at 1/session despite gap satisfied


def test_event_dedup_within_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    breach = mk_state(port_var_95=0.06)
    _script(monkeypatch, [breach, breach])  # same breach two cycles running
    s = fake_settings(tmp_path)
    spy = SpySlack()
    run_engine(
        s,
        max_cycles=2,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 60),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        slack=spy,
        claude_fn=lambda a, b, c: "x",
        config=EngineConfig(event_dedup_window_s=3600.0),
    )
    risk_posts = [m for m in spy.msgs if "risk_breach" in m]
    assert len(risk_posts) == 1  # deduped: persistent condition reported once


def test_claude_gate_persists_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crit = mk_state(port_var_95=0.06)
    s = fake_settings(tmp_path)
    calls: list[int] = []
    common = dict(
        sleep=lambda _x: None,
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        slack=SpySlack(),
        claude_fn=lambda a, b, c: calls.append(1) or "x",
        config=EngineConfig(event_dedup_window_s=0.0),
    )
    _script(monkeypatch, [crit])
    run_engine(s, max_cycles=1, now_fn=_clock(datetime(2026, 6, 3, 14, 0, tzinfo=UTC), 1), **common)
    _script(monkeypatch, [crit])
    # a fresh process 5 min later: persisted last_claude_at still inside the 1800s gap
    run_engine(s, max_cycles=1, now_fn=_clock(datetime(2026, 6, 3, 14, 5, tzinfo=UTC), 1), **common)
    assert len(calls) == 1  # second run respected the persisted min-gap


def test_failsafe_continues_on_cycle_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def flaky_build(*a: object, **k: object):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cycle boom")
        return mk_state()

    monkeypatch.setattr(lp, "build_market_state", flaky_build)
    states = run_engine(
        fake_settings(tmp_path),
        max_cycles=2,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 46),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
    )
    assert len(states) == 1  # first cycle raised + was swallowed; second succeeded


def test_summarize_events_failopen_without_api_key(tmp_path: Path) -> None:
    from quant.engine.events import EngineEvent
    from quant.engine.loop import summarize_events

    ev = [EngineEvent(code="risk_breach", severity="critical", detail="VaR95 6%", at="t")]
    out = summarize_events(mk_state(), ev, fake_settings(tmp_path))
    assert "risk_breach" in out  # deterministic template, no API call


def test_fundamentals_fn_result_reaches_build_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_build(*_a: object, **k: object):
        captured.update(k)
        return mk_state()

    monkeypatch.setattr(lp, "build_market_state", fake_build)
    sentinel = object()
    run_engine(
        fake_settings(tmp_path),
        max_cycles=1,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 46),
        positions_fn=lambda: {"GLD": 1},
        equity_fn=lambda: 1_000_000.0,
        fundamentals_fn=lambda _d: sentinel,  # overrides the default reader
        slack=SpySlack(),
    )
    assert captured.get("fundamentals") is sentinel
