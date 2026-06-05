"""Track C engine integration: credit-stress + recession-onset detectors + loop flow."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quant.engine.events import EventConfig, detect_events
from quant.engine.loop import EngineConfig, engine_dir, run_engine
from quant.macro.nowcast import compute_macro_nowcast
from tests.engine.conftest import fake_settings, mk_state

CFG = EventConfig()


def _clock(start: datetime, step: float):
    t = {"v": start - timedelta(seconds=step)}

    def now() -> datetime:
        t["v"] += timedelta(seconds=step)
        return t["v"]

    return now


def test_credit_stress_warn_on_crossing_stress() -> None:
    evs = detect_events(mk_state(hy_oas=4.5), mk_state(hy_oas=5.5), CFG)
    ev = next(e for e in evs if e.code == "credit_stress")
    assert ev.severity == "warn"


def test_credit_stress_critical_past_high() -> None:
    evs = detect_events(mk_state(hy_oas=4.5), mk_state(hy_oas=8.5), CFG)
    assert next(e for e in evs if e.code == "credit_stress").severity == "critical"


def test_credit_stress_no_refire_when_already_stressed() -> None:
    evs = detect_events(mk_state(hy_oas=5.5), mk_state(hy_oas=6.0), CFG)
    assert not any(e.code == "credit_stress" for e in evs)


def test_recession_onset_on_label_to_high() -> None:
    evs = detect_events(
        mk_state(recession_risk_label="elevated"),
        mk_state(recession_risk_label="high", recession_risk=0.7, macro_cycle_label="contraction"),
        CFG,
    )
    ev = next(e for e in evs if e.code == "recession_onset")
    assert ev.severity == "critical"
    assert "contraction" in ev.detail


def test_recession_onset_no_refire_when_already_high() -> None:
    evs = detect_events(
        mk_state(recession_risk_label="high"), mk_state(recession_risk_label="high"), CFG
    )
    assert not any(e.code == "recession_onset" for e in evs)


def test_nowcast_detectors_quiet_on_first_cycle() -> None:
    codes = {
        e.code for e in detect_events(None, mk_state(hy_oas=9.0, recession_risk_label="high"), CFG)
    }
    assert "credit_stress" not in codes
    assert "recession_onset" not in codes


def test_loop_nowcast_throttled_and_flows_into_state(tmp_path: Path) -> None:
    calls = {"n": 0}

    def nowcast_fn(d: date):
        calls["n"] += 1
        return compute_macro_nowcast(
            d, t10y3m=-0.4, hy_oas=4.5, nfci=0.1, claims=230_000, claims_year_low=210_000, sahm=0.2
        )

    run_engine(
        fake_settings(tmp_path),
        max_cycles=3,
        dry_run=True,
        sleep=lambda _x: None,
        now_fn=_clock(datetime(2026, 6, 4, 14, tzinfo=UTC), 46),
        positions_fn=lambda: None,
        equity_fn=lambda: None,
        intraday_fn=lambda: None,
        news_fn=lambda: None,
        eventrisk_fn=lambda _d: None,
        macro_nowcast_fn=nowcast_fn,
        config=EngineConfig(nowcast_refresh_s=1800.0),  # 46s steps -> fetched once
    )
    assert calls["n"] == 1  # throttled across the 3 cycles
    state = json.loads((engine_dir(tmp_path) / "state.json").read_text())
    assert state["macro_cycle_label"] == "late-cycle"
    assert state["hy_oas"] == 4.5
    assert state["term_spread_10y3m"] == -0.4
