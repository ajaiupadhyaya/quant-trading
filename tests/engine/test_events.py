"""Deterministic event detectors: each material change fires the right event."""

from __future__ import annotations

from quant.engine.events import EventConfig, detect_events
from tests.engine.conftest import mk_state

CFG = EventConfig()


def _codes(prev, curr, **kw):
    return {e.code for e in detect_events(prev, curr, CFG, **kw)}


def test_no_change_emits_nothing() -> None:
    assert detect_events(mk_state(), mk_state(), CFG) == []


def test_regime_flip_to_crisis_is_critical() -> None:
    evs = detect_events(mk_state(regime_label="calm"), mk_state(regime_label="crisis"), CFG)
    e = next(e for e in evs if e.code == "regime_flip")
    assert e.severity == "critical"


def test_posture_cross_to_risk_off_is_critical() -> None:
    evs = detect_events(
        mk_state(composite_label="neutral"), mk_state(composite_label="risk-off"), CFG
    )
    e = next(e for e in evs if e.code == "posture_cross")
    assert e.severity == "critical"


def test_posture_cross_to_neutral_is_warn() -> None:
    evs = detect_events(
        mk_state(composite_label="risk-off"), mk_state(composite_label="neutral"), CFG
    )
    e = next(e for e in evs if e.code == "posture_cross")
    assert e.severity == "warn"


def test_vol_spike_on_jump_and_on_absolute_level() -> None:
    assert "vol_spike" in _codes(mk_state(vix=16.0), mk_state(vix=22.0))  # +6 jump
    crit = detect_events(mk_state(vix=16.0), mk_state(vix=30.0, vol_regime="stressed"), CFG)
    assert next(e for e in crit if e.code == "vol_spike").severity == "critical"


def test_breadth_collapse() -> None:
    assert "breadth_collapse" in _codes(mk_state(breadth=0.8), mk_state(breadth=0.2))  # absolute low
    assert "breadth_collapse" in _codes(mk_state(breadth=0.9), mk_state(breadth=0.6))  # -0.3 drop


def test_corr_spike() -> None:
    assert "corr_spike" in _codes(mk_state(avg_corr=0.2), mk_state(avg_corr=0.4))
    assert "corr_spike" not in _codes(mk_state(avg_corr=0.2), mk_state(avg_corr=0.25))


def test_risk_breach_is_critical() -> None:
    evs = detect_events(mk_state(), mk_state(port_var_95=0.06), CFG)
    e = next(e for e in evs if e.code == "risk_breach")
    assert e.severity == "critical"
    assert "VaR95" in e.detail


def test_halt_onset_is_critical() -> None:
    evs = detect_events(mk_state(halt_active=False), mk_state(halt_active=True), CFG)
    assert next(e for e in evs if e.code == "halt").severity == "critical"
    # an already-active halt that didn't just flip does not re-fire
    assert "halt" not in _codes(mk_state(halt_active=True), mk_state(halt_active=True))


def test_drawdown_uses_session_high() -> None:
    evs = detect_events(
        mk_state(equity=1_000_000.0),
        mk_state(equity=960_000.0),
        CFG,
        session_high_equity=1_000_000.0,
    )
    e = next(e for e in evs if e.code == "drawdown")
    assert e.severity == "warn"  # -4% is between -3% and -6%
    deep = detect_events(
        mk_state(), mk_state(equity=920_000.0), CFG, session_high_equity=1_000_000.0
    )
    assert next(e for e in deep if e.code == "drawdown").severity == "critical"  # -8%


def test_first_cycle_fires_absolute_but_not_transitions() -> None:
    # prev=None: absolute danger surfaces, fabricated 'transitions' do not.
    codes = _codes(None, mk_state(vix=30.0, port_var_95=0.06, composite_label="risk-off"))
    assert "vol_spike" in codes and "risk_breach" in codes  # absolute
    assert "posture_cross" not in codes and "regime_flip" not in codes  # transitions suppressed
    assert detect_events(None, mk_state(), CFG) == []  # calm startup = silent
