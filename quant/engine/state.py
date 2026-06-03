"""MarketState: a flat, serializable, diff-able snapshot of the live quant brain.

Built by composing the proven read-only readers (``gather_analyst_context`` —
signals + regime + macro + governance + portfolio risk) into scalar fields the
event bus can compare cycle-to-cycle, plus the monitor halt/severity and the
session phase. Every field is best-effort: a missing source degrades to ``None``
and is recorded in ``degraded``; building a state NEVER raises.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from quant.deploy.calendar_clock import is_trading_day, session_close_et, to_et
from quant.util.logging import logger

_PREMARKET_START = time(7, 0)
_MARKET_OPEN = time(9, 30)
_AFTERHOURS_END = time(20, 0)


@dataclass(frozen=True)
class MarketState:
    """One cycle's read-only snapshot of the market + the live book."""

    at: str  # UTC ISO timestamp
    asof: str  # ET trading date (ISO)
    session_phase: str  # "closed" | "premarket" | "rth" | "afterhours"
    is_trading_day: bool
    # composite posture + signals
    composite_score: float | None
    composite_label: str | None
    coverage: float | None
    breadth: float | None
    median_mom: float | None
    avg_corr: float | None
    # regime
    regime_label: str | None
    p_crisis: float | None
    # volatility
    vix: float | None
    realized_vol: float | None
    vol_regime: str | None
    # rates
    curve_label: str | None
    term_spread: float | None
    # live book / portfolio risk
    equity: float | None
    n_positions: int | None
    port_ann_vol: float | None
    port_var_95: float | None
    port_cvar_95: float | None
    port_beta: float | None
    top_name_weight: float | None
    # monitor / governance
    halt_active: bool
    worst_severity: str | None
    live_strategies: tuple[str, ...]
    degraded: tuple[str, ...]


def session_phase(now_utc: datetime, asof: date) -> str:
    """Classify the current instant into a coarse session phase (ET)."""
    if not is_trading_day(asof):
        return "closed"
    et = to_et(now_utc).time()
    close = session_close_et(asof)
    if et < _PREMARKET_START or et >= _AFTERHOURS_END:
        return "closed"
    if et < _MARKET_OPEN:
        return "premarket"
    if et < close:
        return "rth"
    return "afterhours"


def _f(x: Any) -> float | None:
    """Coerce to a FINITE float, else None. Non-finite (NaN/inf) -> None so every
    persisted/serialized state field is valid JSON and the event detectors never
    see NaN (which would defeat comparison guards)."""
    try:
        v = float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and math.isfinite(v)) else None


def _read_monitor(data_dir: Path) -> tuple[bool, str | None]:
    """halt_active + worst_severity from the guard's status artifact. Fail-open."""
    try:
        from quant.monitor.status import monitor_status_path

        path = monitor_status_path(data_dir)
        if not path.exists():
            return False, None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False, None
        return bool(payload.get("halt_active", False)), (
            str(payload["worst_severity"]) if payload.get("worst_severity") is not None else None
        )
    except Exception as exc:  # fail-open
        logger.info("engine.state: monitor read skipped ({!r})", exc)
        return False, None


def build_market_state(
    data_dir: Path,
    *,
    asof: date,
    now_utc: datetime | None = None,
    positions: dict[str, int] | None = None,
    equity: float | None = None,
) -> MarketState:
    """Compose one read-only MarketState. Best-effort end to end; never raises."""
    now = now_utc or datetime.now(UTC)
    degraded: list[str] = []

    ctx: Any | None = None
    try:
        from quant.analyst.context import gather_analyst_context

        ctx = gather_analyst_context(
            data_dir, asof, include_macro=True, positions=positions, equity=equity
        )
    except Exception as exc:  # fail-open: a degraded state is still useful
        logger.info("engine.state: context gather skipped ({!r})", exc)
        degraded.append("context")

    sig = getattr(ctx, "signals", None)
    agg = getattr(sig, "aggregates", None)
    vol = getattr(sig, "vol", None)
    corr = getattr(sig, "corr", None)
    rates = getattr(sig, "rates", None)
    regime = getattr(ctx, "regime", None)
    prisk = getattr(ctx, "portfolio_risk", None)
    macro = getattr(ctx, "macro", {}) or {}

    if sig is None:
        degraded.append("signals")
    if regime is None:
        degraded.append("regime")
    if prisk is None:
        degraded.append("portfolio_risk")

    # Live strategies = governance evidence with state == "live".
    live_strats: list[str] = []
    for ev in getattr(ctx, "evidence", []) or []:
        if str(getattr(ev, "state", "") or "").lower() == "live":
            live_strats.append(str(getattr(ev, "slug", "")))

    halt_active, worst_severity = _read_monitor(data_dir)

    vix = _f(getattr(vol, "vix_level", None))
    if vix is None:
        vix = _f(macro.get("vix"))

    return MarketState(
        at=now.replace(microsecond=0).isoformat(),
        asof=asof.isoformat(),
        session_phase=session_phase(now, asof),
        is_trading_day=is_trading_day(asof),
        composite_score=_f(getattr(sig, "composite_score", None)),
        composite_label=getattr(sig, "composite_label", None),
        coverage=_f(getattr(sig, "coverage", None)),
        breadth=_f(getattr(agg, "breadth_above_trend", None)),
        median_mom=_f(getattr(agg, "median_mom_blended", None)),
        avg_corr=_f(getattr(corr, "avg_pairwise_corr", None)),
        regime_label=getattr(regime, "label", None),
        p_crisis=_f(getattr(regime, "p_crisis", None)),
        vix=vix,
        realized_vol=_f(getattr(vol, "spy_realized_vol_ann", None)),
        vol_regime=getattr(vol, "vol_regime", None),
        curve_label=getattr(rates, "curve_label", None),
        term_spread=_f(getattr(rates, "term_spread", None)),
        equity=_f(equity),
        n_positions=(len(positions) if positions is not None else None),
        port_ann_vol=_f(getattr(prisk, "ann_vol", None)),
        port_var_95=_f(getattr(prisk, "var_95", None)),
        port_cvar_95=_f(getattr(prisk, "cvar_95", None)),
        port_beta=_f(getattr(prisk, "beta_to_benchmark", None)),
        top_name_weight=_f(getattr(prisk, "top_name_weight", None)),
        halt_active=halt_active,
        worst_severity=worst_severity,
        live_strategies=tuple(s for s in live_strats if s),
        degraded=tuple(degraded),
    )


def to_json_dict(state: MarketState) -> dict[str, Any]:
    payload = asdict(state)
    payload["live_strategies"] = list(state.live_strategies)
    payload["degraded"] = list(state.degraded)
    return payload


def from_json_dict(payload: dict[str, Any]) -> MarketState:
    data = dict(payload)
    data["live_strategies"] = tuple(data.get("live_strategies", []) or [])
    data["degraded"] = tuple(data.get("degraded", []) or [])
    kwargs: dict[str, Any] = {k: data.get(k) for k in MarketState.__dataclass_fields__}
    return MarketState(**kwargs)


def render_state(state: MarketState) -> str:
    """Terse one-liner for the CLI + Slack + logs."""
    bits = [f"[{state.session_phase}] {state.at}"]
    if state.composite_label:
        sc = f"{state.composite_score:+.2f}" if state.composite_score is not None else "n/a"
        bits.append(f"posture={state.composite_label}({sc})")
    if state.regime_label:
        bits.append(f"regime={state.regime_label}")
    if state.vix is not None:
        bits.append(f"vix={state.vix:.1f}")
    if state.vol_regime:
        bits.append(f"vol={state.vol_regime}")
    if state.equity is not None:
        bits.append(f"equity=${state.equity:,.0f}")
    if state.port_var_95 is not None:
        bits.append(f"VaR95={state.port_var_95:.2%}")
    if state.halt_active:
        bits.append("HALT")
    if state.degraded:
        bits.append(f"degraded={','.join(state.degraded)}")
    return " | ".join(bits)
