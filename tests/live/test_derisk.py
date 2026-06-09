"""Deterministic one-way de-risk overlay: multiplier math, fail-safe direction, shadow gate."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from quant.live.derisk import DeriskConfig, derisk_multiplier, to_report_dict

_NOW = datetime(2026, 6, 9, 18, 0, tzinfo=UTC)


def _state(at: datetime = _NOW, **fields: Any) -> dict[str, Any]:
    s: dict[str, Any] = {"at": at.replace(microsecond=0).isoformat()}
    s.update(fields)
    return s


def test_no_risk_signals_is_neutral() -> None:
    r = derisk_multiplier(_state(composite_label="risk-on", vol_regime="normal"), DeriskConfig(), now=_NOW)
    assert r.multiplier == 1.0
    assert r.applied == 1.0
    assert r.reasons == []
    assert r.degraded is False


def test_risk_off_reduces_multiplier_but_shadow_does_not_apply() -> None:
    cfg = DeriskConfig()  # actuate=False (default)
    r = derisk_multiplier(_state(composite_label="risk-off"), cfg, now=_NOW)
    assert r.multiplier == 1.0 - cfg.w_risk_off  # 0.75 computed
    assert r.applied == 1.0  # SHADOW: not applied
    assert r.actuated is False
    assert any("risk-off" in x for x in r.reasons)


def test_actuate_applies_the_multiplier() -> None:
    cfg = DeriskConfig(actuate=True)
    r = derisk_multiplier(_state(composite_label="risk-off"), cfg, now=_NOW)
    assert r.applied == r.multiplier == 1.0 - cfg.w_risk_off
    assert r.actuated is True


def test_multiple_signals_stack_and_clamp_to_floor() -> None:
    state = _state(
        composite_label="risk-off",
        regime_label="crisis",
        vol_regime="stressed",
        hy_oas=8.0,  # percent: 8% HY OAS = acute credit stress (>= 5.0 gate)
        recession_risk_label="high",
        intraday_spy_ret=-0.03,
    )
    r = derisk_multiplier(state, DeriskConfig(actuate=True, floor=0.5), now=_NOW)
    # reduction 0.25+0.20+0.15+0.15+0.15+0.15 = 1.05 -> clamp to floor 0.5
    assert r.multiplier == 0.5
    assert r.applied == 0.5
    assert len(r.reasons) == 6


def test_one_way_never_above_one() -> None:
    state = _state(composite_label="risk-on", hy_oas=3.0, intraday_spy_ret=0.02)  # 3% HY = benign
    r = derisk_multiplier(state, DeriskConfig(actuate=True), now=_NOW)
    assert r.multiplier == 1.0  # benign signals can only keep it at 1.0, never raise it


def test_stale_state_never_causes_derisk() -> None:
    state = _state(at=_NOW - timedelta(hours=5), composite_label="risk-off")  # risk-off BUT stale
    r = derisk_multiplier(state, DeriskConfig(actuate=True, max_staleness_minutes=120), now=_NOW)
    assert r.degraded is True
    assert r.applied == 1.0 and r.multiplier == 1.0  # stale fails to the safe (no-derisk) side


def test_missing_state_never_causes_derisk() -> None:
    r = derisk_multiplier(None, DeriskConfig(actuate=True), now=_NOW)
    assert r.degraded is True
    assert r.applied == 1.0


def test_degenerate_numeric_fields_ignored() -> None:
    state = _state(composite_label="risk-on", hy_oas="bad", intraday_spy_ret=None)
    r = derisk_multiplier(state, DeriskConfig(actuate=True), now=_NOW)
    assert r.multiplier == 1.0  # non-numeric fields contribute no de-risk


def test_to_report_dict_is_json_serializable() -> None:
    r = derisk_multiplier(_state(composite_label="risk-off"), DeriskConfig(), now=_NOW)
    d = to_report_dict(r)
    assert d["multiplier"] == r.multiplier and d["applied"] == 1.0 and d["actuated"] is False
    assert isinstance(d["reasons"], list)
    assert json.loads(json.dumps(d)) == d
