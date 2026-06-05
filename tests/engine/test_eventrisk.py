"""Phase 7C engine integration: event-risk transition detectors + loop throttle."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quant.engine.events import EventConfig, detect_events
from quant.engine.loop import EngineConfig, engine_dir, run_engine
from quant.macro.events import EventRisk
from tests.engine.conftest import fake_settings, mk_state

CFG = EventConfig()


def test_event_window_fires_on_entering_only() -> None:
    enter = detect_events(
        mk_state(in_event_window=False),
        mk_state(in_event_window=True, next_event="FOMC", days_to_event=2),
        CFG,
    )
    assert next(e for e in enter if e.code == "event_window").severity == "warn"
    # already in the window (no transition) -> no repeat
    assert not any(
        e.code == "event_window"
        for e in detect_events(mk_state(in_event_window=True), mk_state(in_event_window=True), CFG)
    )


def test_vix_backwardation_is_critical_on_inversion() -> None:
    evs = detect_events(mk_state(vix_term_structure=1.10), mk_state(vix_term_structure=0.95), CFG)
    assert next(e for e in evs if e.code == "vix_backwardation").severity == "critical"
    # staying inverted (no fresh crossing) does not re-fire
    assert not any(
        e.code == "vix_backwardation"
        for e in detect_events(
            mk_state(vix_term_structure=0.95), mk_state(vix_term_structure=0.93), CFG
        )
    )


def test_financial_conditions_tighten_crossing_zero() -> None:
    evs = detect_events(
        mk_state(financial_conditions=-0.10), mk_state(financial_conditions=0.20), CFG
    )
    assert any(e.code == "financial_conditions_tighten" for e in evs)
    assert not any(
        e.code == "financial_conditions_tighten"
        for e in detect_events(
            mk_state(financial_conditions=-0.30), mk_state(financial_conditions=-0.20), CFG
        )
    )


def test_event_detectors_quiet_on_first_cycle() -> None:
    # prev=None: transitions must not fire even if currently in a window / inverted.
    codes = {
        e.code
        for e in detect_events(
            None,
            mk_state(in_event_window=True, vix_term_structure=0.9, financial_conditions=0.3),
            CFG,
        )
    }
    assert "event_window" not in codes
    assert "vix_backwardation" not in codes
    assert "financial_conditions_tighten" not in codes


def _clock(start: datetime, step: float):
    t = {"v": start - timedelta(seconds=step)}

    def now() -> datetime:
        t["v"] += timedelta(seconds=step)
        return t["v"]

    return now


def _er(label: str = "watch") -> EventRisk:
    return EventRisk(
        next_event="Jobs report",
        next_event_date="2026-06-05",
        days_to_event=2,
        in_event_window=True,
        policy_uncertainty=329.0,
        policy_uncertainty_elevated=True,
        financial_conditions=-0.49,
        financial_stress=-0.69,
        vix_term_structure=1.21,
        risk_label=label,
    )


def test_loop_eventrisk_throttled_and_flows_into_state(tmp_path: Path) -> None:
    calls = {"n": 0}

    def er_fn(d: date):
        calls["n"] += 1
        return _er()

    run_engine(
        fake_settings(tmp_path),
        max_cycles=3,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 3, 14, tzinfo=UTC), 46),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        intraday_fn=lambda: None,
        news_fn=lambda: None,
        eventrisk_fn=er_fn,
        config=EngineConfig(eventrisk_refresh_s=1800.0),  # 46s steps -> fetched once
    )
    assert calls["n"] == 1
    state = json.loads((engine_dir(tmp_path) / "state.json").read_text())
    assert state["macro_risk_label"] == "watch"
    assert state["next_event"] == "Jobs report" and state["in_event_window"] is True
    assert abs(state["policy_uncertainty"] - 329.0) < 1e-9
