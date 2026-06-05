"""Phase 7C event calendar + macro/policy risk read."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from quant.macro import events as ev
from quant.macro.events import (
    EventRiskConfig,
    compute_event_risk,
    live_event_risk,
    next_high_impact_event,
    render_event_risk,
    upcoming_events,
)


def test_calendar_computations() -> None:
    assert ev._nfp(2026, 6) == date(2026, 6, 5)  # first Friday
    assert ev._opex(2026, 6) == date(2026, 6, 19)  # third Friday (quad-witching month)
    assert ev._election_day(2026) == date(2026, 11, 3)  # Tue after first Mon in Nov
    assert ev._election_day(2025) is None  # odd year, no federal election
    qend = ev._last_business_day(2026, 6)
    assert qend.month == 6 and qend.weekday() < 5  # a weekday


def test_next_high_impact_and_upcoming() -> None:
    asof = date(2026, 6, 3)
    nxt = next_high_impact_event(asof)
    assert nxt is not None and nxt.name == "Jobs report" and nxt.date == "2026-06-05"
    names = {e.name for e in upcoming_events(asof, horizon_days=21)}
    assert "FOMC" in names  # embedded 2026-06-17 is in the window
    assert "Quad-witching" in names


def test_compute_event_risk_window_and_labels() -> None:
    asof = date(2026, 6, 3)  # 2 days before the jobs report
    r = compute_event_risk(asof, epu=329, nfci=-0.49, finstress=-0.69, vix=16.0, vix3m=19.5)
    assert r.in_event_window is True and r.days_to_event == 2
    assert r.policy_uncertainty_elevated is True  # EPU 329 > 200
    assert abs(r.vix_term_structure - 19.5 / 16.0) < 1e-9
    assert r.risk_label == "watch"  # event window + elevated EPU, but no acute stress


def test_compute_event_risk_stressed_on_backwardation() -> None:
    r = compute_event_risk(date(2026, 7, 1), vix=30.0, vix3m=27.0)  # VXV/VIX = 0.9 < 1
    assert r.vix_term_structure is not None and r.vix_term_structure < 1.0
    assert r.risk_label == "stressed"


def test_compute_event_risk_calm_when_quiet() -> None:
    # A date far from any high-impact event, loose conditions, contango.
    r = compute_event_risk(
        date(2026, 8, 12), epu=90, nfci=-0.6, finstress=-0.8, vix=14.0, vix3m=16.0
    )
    assert r.in_event_window is False
    assert r.risk_label == "calm"


def test_missing_inputs_degrade_gracefully() -> None:
    r = compute_event_risk(date(2026, 8, 12))  # no FRED inputs at all
    assert r.policy_uncertainty is None and r.vix_term_structure is None
    assert r.next_event is not None  # calendar still works
    assert r.risk_label in {"calm", "watch"}


def test_live_event_risk_failopen(monkeypatch: pytest.MonkeyPatch) -> None:
    # FRED down: uncertainty fields None, but the calendar still computes; never raises.
    from quant.data import macro as macro_mod

    monkeypatch.setattr(macro_mod, "get_series", lambda code: (_ for _ in ()).throw(RuntimeError()))
    r = live_event_risk(SimpleNamespace(), date(2026, 6, 3), config=EventRiskConfig())
    assert r.policy_uncertainty is None
    assert r.next_event is not None  # calendar independent of FRED


def test_render_event_risk() -> None:
    assert render_event_risk(None) == "Event risk: unavailable"
    r = compute_event_risk(date(2026, 6, 3), epu=329, nfci=-0.4, vix=16.0, vix3m=19.5)
    out = render_event_risk(r)
    assert "macro-risk=watch" in out and "EPU=329" in out and "WINDOW" in out
