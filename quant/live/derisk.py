"""Deterministic one-way de-risk overlay for the live rebalance.

Reads the continuous engine's MarketState (``data/engine/state.json``) and derives a bounded
risk-posture MULTIPLIER in ``[floor, 1.0]`` from deterministic risk-off signals (composite
posture, HMM regime, vol regime, credit/recession, intraday drawdown). The rebalance scales
each strategy's equity slice by this factor, so the overlay can ONLY SHRINK gross exposure;
a later rebalance with no risk-off signal restores full size (fully reversible). This is the
deterministic Tier-0 link that lets the engine's risk-off detection actually de-risk the book
instead of only logging — within the existing halt + Guard-4/5 + governance envelope.

Contract:
- ``actuate=False`` (default) ⇒ SHADOW: the multiplier is computed and reported but NOT
  applied; the rebalance is byte-for-byte today's behavior. Turning actuation on is a
  deliberate, reversible human flip.
- Fail-SAFE direction: any missing / degraded / stale input contributes NO de-risk
  (multiplier stays 1.0). A stale or unreadable engine state can never *cause* de-risking.
- Pure + deterministic: ``derisk_multiplier`` does no I/O; ``load_engine_state`` is the only
  read and is itself fail-open.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant.util.logging import logger


@dataclass(frozen=True)
class DeriskConfig:
    """Weights and gates for the deterministic de-risk overlay. Inert by default:
    ``actuate=False`` makes the overlay a pure shadow (compute + report, apply nothing)."""

    actuate: bool = False
    floor: float = 0.5  # never de-risk below this fraction of full gross (one rebalance step)
    max_staleness_minutes: float = 120.0  # older engine state ⇒ degraded ⇒ no de-risk
    # Each adverse signal subtracts its weight from a full 1.0 multiplier (then clamped to floor).
    w_risk_off: float = 0.25  # composite posture == "risk-off"
    # HMM regime "crisis" is UNVALIDATED + miscalibrated today (confident crisis on calm-vol
    # days, label flips with the training window), so it is EXCLUDED by default (weight 0).
    # Raise this only after the regime model passes its OOS gates (`quant regime validate`).
    w_regime_crisis: float = 0.0
    w_vol_stressed: float = 0.15  # vol_regime == "stressed"
    w_credit_stress: float = 0.15  # HY OAS >= hy_oas_stress
    w_recession: float = 0.15  # recession_risk_label elevated/high
    w_intraday_drawdown: float = 0.15  # intraday SPY return <= intraday_drawdown_threshold
    hy_oas_stress: float = 5.0  # HY OAS in PERCENT (matches quant.macro.nowcast: 5% = stress)
    intraday_drawdown_threshold: float = -0.015  # SPY intraday <= -1.5%


@dataclass(frozen=True)
class DeriskResult:
    multiplier: float  # the computed one-way factor in [floor, 1.0]
    applied: float  # what the rebalance actually uses: multiplier if actuated else 1.0
    actuated: bool
    reasons: list[str] = field(default_factory=list)
    state_at: str | None = None  # engine-state timestamp the decision was read from
    degraded: bool = False  # state missing / unreadable / stale ⇒ no de-risk


def load_engine_state(data_dir: Path) -> dict[str, Any] | None:
    """Best-effort read of the engine's latest MarketState; None if absent/unreadable."""
    path = data_dir / "engine" / "state.json"
    if not path.exists():
        return None
    try:
        loaded: dict[str, Any] = json.loads(path.read_text())
        return loaded
    except Exception as exc:  # fail-open: a bad engine state must never break a rebalance
        logger.info("derisk: engine state unreadable ({!r})", exc)
        return None


def _stale(state_at: Any, max_minutes: float, now: datetime) -> bool:
    if not isinstance(state_at, str) or not state_at:
        return True
    try:
        ts = datetime.fromisoformat(state_at)
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (now - ts).total_seconds() > max_minutes * 60.0


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def derisk_multiplier(
    state: dict[str, Any] | None, cfg: DeriskConfig, *, now: datetime | None = None
) -> DeriskResult:
    """Deterministic one-way de-risk factor in ``[cfg.floor, 1.0]`` from the engine MarketState.

    Each adverse risk-off signal subtracts its weight from 1.0; the result is clamped to
    ``cfg.floor`` (never below). Missing/degraded fields contribute nothing — the only
    failure direction is *less* de-risk, never more. ``applied`` is the computed multiplier
    when ``cfg.actuate`` is True, else 1.0 (shadow)."""
    current = now if now is not None else datetime.now(UTC)
    if state is None:
        return DeriskResult(1.0, 1.0, cfg.actuate, ["no-engine-state"], None, True)
    state_at = state.get("at")
    if _stale(state_at, cfg.max_staleness_minutes, current):
        return DeriskResult(1.0, 1.0, cfg.actuate, ["engine-state-stale"], state_at, True)

    reduction = 0.0
    reasons: list[str] = []

    # Each signal contributes only when its weight is > 0, so a zeroed weight is genuinely
    # off (e.g. the unvalidated regime signal) rather than a no-op reason at -0.00.
    if cfg.w_risk_off > 0.0 and state.get("composite_label") == "risk-off":
        reduction += cfg.w_risk_off
        reasons.append(f"posture=risk-off(-{cfg.w_risk_off:.2f})")
    if cfg.w_regime_crisis > 0.0 and "crisis" in str(state.get("regime_label") or "").lower():
        reduction += cfg.w_regime_crisis
        reasons.append(f"regime=crisis(-{cfg.w_regime_crisis:.2f})")
    if cfg.w_vol_stressed > 0.0 and state.get("vol_regime") == "stressed":
        reduction += cfg.w_vol_stressed
        reasons.append(f"vol=stressed(-{cfg.w_vol_stressed:.2f})")
    hy = _num(state.get("hy_oas"))
    if cfg.w_credit_stress > 0.0 and hy is not None and hy >= cfg.hy_oas_stress:
        reduction += cfg.w_credit_stress
        reasons.append(f"credit=stress(hy_oas={hy:.3f},-{cfg.w_credit_stress:.2f})")
    if cfg.w_recession > 0.0 and str(state.get("recession_risk_label") or "").lower() in (
        "elevated",
        "high",
    ):
        reduction += cfg.w_recession
        reasons.append(f"recession={state.get('recession_risk_label')}(-{cfg.w_recession:.2f})")
    intraday = _num(state.get("intraday_spy_ret"))
    if cfg.w_intraday_drawdown > 0.0 and intraday is not None and (
        intraday <= cfg.intraday_drawdown_threshold
    ):
        reduction += cfg.w_intraday_drawdown
        reasons.append(f"intraday_dd={intraday:.3f}(-{cfg.w_intraday_drawdown:.2f})")

    multiplier = max(cfg.floor, round(1.0 - reduction, 4))
    applied = multiplier if cfg.actuate else 1.0
    return DeriskResult(multiplier, applied, cfg.actuate, reasons, state_at, False)


def to_report_dict(result: DeriskResult) -> dict[str, Any]:
    """Serializable shadow payload for the rebalance report / artifact."""
    return {
        "multiplier": result.multiplier,
        "applied": result.applied,
        "actuated": result.actuated,
        "reasons": list(result.reasons),
        "engine_state_at": result.state_at,
        "degraded": result.degraded,
    }
