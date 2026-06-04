"""Deterministic event bus: detect MATERIAL changes between two MarketStates.

These pure functions are the engine's autonomous judgment — the quant layer
deciding what is worth surfacing, with NO Claude in the loop. The continuous
loop posts every event to Slack/log and escalates only the highest-severity ones
to Claude (rate-limited). An event NEVER triggers a trade or a halt; it is a
notification + an input to the (separate, human-gated) actuation phases.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant.engine.state import MarketState

_SEVERITY_ORDER = {"info": 0, "warn": 1, "critical": 2}


def severity_at_least(severity: str, threshold: str) -> bool:
    return _SEVERITY_ORDER.get(severity, 0) >= _SEVERITY_ORDER.get(threshold, 99)


@dataclass(frozen=True)
class EngineEvent:
    code: str
    severity: str  # "info" | "warn" | "critical"
    detail: str
    at: str


@dataclass(frozen=True)
class EventConfig:
    """Thresholds for the deterministic detectors. Deliberately conservative so
    the engine surfaces genuine regime/risk shifts, not routine noise."""

    vix_spike_abs: float = 5.0  # +pts cycle-over-cycle
    vix_stress_level: float = 28.0
    breadth_collapse_level: float = 0.30
    breadth_drop_abs: float = 0.25
    avg_corr_spike_abs: float = 0.15
    intraday_drawdown_pct: float = -0.03  # vs session-high equity
    port_var_limit: float = 0.05
    port_cvar_limit: float = 0.07
    port_vol_limit: float = 0.35
    port_beta_limit: float = 1.50
    # intraday (Phase 7A): live within-session moves
    intraday_selloff_warn: float = -0.015  # SPY down on the day
    intraday_selloff_crit: float = -0.03
    intraday_breadth_break: float = 0.20  # share of the universe up — a real washout
    intraday_range_warn: float = 0.30  # SPY Parkinson range-vol (annualized)
    intraday_range_crit: float = 0.50
    # news sentiment (Phase 7B): a genuinely negative headline environment
    news_negative_warn: float = -0.30
    news_negative_crit: float = -0.50
    news_min_items: int = 3  # need a real batch before flagging
    # macro / event risk (Phase 7C) — transition-based to avoid multi-day spam
    nfci_tight: float = 0.0  # NFCI crossing above 0 = conditions tightening
    vix_backwardation: float = 1.0  # VXV/VIX crossing below 1 = near-term stress
    # macro / cycle nowcast (track C) — transition-based
    hy_oas_stress: float = 5.0  # HY OAS (percent) crossing above this = credit stress
    hy_oas_high: float = 8.0  # acute credit stress


def _regime_flip(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if not prev.regime_label or not curr.regime_label or prev.regime_label == curr.regime_label:
        return None
    to_crisis = "crisis" in curr.regime_label.lower()
    return EngineEvent(
        code="regime_flip",
        severity="critical" if to_crisis else "warn",
        detail=f"regime {prev.regime_label} -> {curr.regime_label}",
        at=curr.at,
    )


def _posture_cross(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if not prev.composite_label or not curr.composite_label:
        return None
    if prev.composite_label == curr.composite_label:
        return None
    to_risk_off = curr.composite_label == "risk-off"
    return EngineEvent(
        code="posture_cross",
        severity="critical" if to_risk_off else "warn",
        detail=f"posture {prev.composite_label} -> {curr.composite_label}",
        at=curr.at,
    )


def _vol_spike(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if curr.vix is None:
        return None
    jumped = prev.vix is not None and (curr.vix - prev.vix) >= cfg.vix_spike_abs
    stressed = (
        curr.vol_regime == "stressed" and prev.vol_regime != "stressed"
    ) or curr.vix >= cfg.vix_stress_level
    if not (jumped or stressed):
        return None
    sev = (
        "critical"
        if (curr.vix >= cfg.vix_stress_level or curr.vol_regime == "stressed")
        else "warn"
    )
    delta = f" (+{curr.vix - prev.vix:.1f})" if prev.vix is not None else ""
    return EngineEvent(
        code="vol_spike", severity=sev, detail=f"VIX {curr.vix:.1f}{delta}", at=curr.at
    )


def _breadth_collapse(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if curr.breadth is None:
        return None
    low = curr.breadth <= cfg.breadth_collapse_level
    dropped = prev.breadth is not None and (prev.breadth - curr.breadth) >= cfg.breadth_drop_abs
    if not (low or dropped):
        return None
    return EngineEvent(
        code="breadth_collapse",
        severity="warn",
        detail=f"breadth {curr.breadth:.0%} above 200d",
        at=curr.at,
    )


def _corr_spike(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if curr.avg_corr is None or prev.avg_corr is None:
        return None
    if (curr.avg_corr - prev.avg_corr) < cfg.avg_corr_spike_abs:
        return None
    return EngineEvent(
        code="corr_spike",
        severity="warn",
        detail=f"avg corr {prev.avg_corr:+.2f} -> {curr.avg_corr:+.2f} (fragility)",
        at=curr.at,
    )


def _risk_breach(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    breaches: list[str] = []
    if curr.port_var_95 is not None and curr.port_var_95 > cfg.port_var_limit:
        breaches.append(f"VaR95 {curr.port_var_95:.2%}>{cfg.port_var_limit:.0%}")
    if curr.port_cvar_95 is not None and curr.port_cvar_95 > cfg.port_cvar_limit:
        breaches.append(f"CVaR95 {curr.port_cvar_95:.2%}>{cfg.port_cvar_limit:.0%}")
    if curr.port_ann_vol is not None and curr.port_ann_vol > cfg.port_vol_limit:
        breaches.append(f"vol {curr.port_ann_vol:.0%}>{cfg.port_vol_limit:.0%}")
    if curr.port_beta is not None and abs(curr.port_beta) > cfg.port_beta_limit:
        breaches.append(f"|beta| {abs(curr.port_beta):.2f}>{cfg.port_beta_limit:.2f}")
    if not breaches:
        return None
    return EngineEvent(
        code="risk_breach",
        severity="critical",
        detail="portfolio risk: " + ", ".join(breaches),
        at=curr.at,
    )


def _intraday_selloff(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if curr.intraday_spy_ret is None or curr.intraday_spy_ret > cfg.intraday_selloff_warn:
        return None
    sev = "critical" if curr.intraday_spy_ret <= cfg.intraday_selloff_crit else "warn"
    return EngineEvent(
        code="intraday_selloff",
        severity=sev,
        detail=f"SPY {curr.intraday_spy_ret:+.2%} on the day",
        at=curr.at,
    )


def _intraday_breadth_break(
    prev: MarketState, curr: MarketState, cfg: EventConfig
) -> EngineEvent | None:
    if curr.intraday_breadth is None or curr.intraday_breadth > cfg.intraday_breadth_break:
        return None
    return EngineEvent(
        code="intraday_breadth_break",
        severity="warn",
        detail=f"only {curr.intraday_breadth:.0%} of the universe up today (broad selloff)",
        at=curr.at,
    )


def _intraday_range_spike(
    prev: MarketState, curr: MarketState, cfg: EventConfig
) -> EngineEvent | None:
    if curr.intraday_range_vol is None or curr.intraday_range_vol < cfg.intraday_range_warn:
        return None
    sev = "critical" if curr.intraday_range_vol >= cfg.intraday_range_crit else "warn"
    return EngineEvent(
        code="intraday_range_spike",
        severity=sev,
        detail=f"SPY intraday range-vol {curr.intraday_range_vol:.0%} annualized",
        at=curr.at,
    )


def _negative_news(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if curr.news_sentiment is None or (curr.news_n_items or 0) < cfg.news_min_items:
        return None
    if curr.news_sentiment > cfg.news_negative_warn:
        return None
    sev = "critical" if curr.news_sentiment <= cfg.news_negative_crit else "warn"
    detail = f"news sentiment {curr.news_sentiment:+.2f} over {curr.news_n_items} items"
    if curr.news_top_negative:
        detail += f' — "{curr.news_top_negative[:80]}"'
    return EngineEvent(code="negative_news", severity=sev, detail=detail, at=curr.at)


def _event_window(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    # Fires once on ENTERING a high-impact event window (FOMC / jobs / quad-witching).
    if curr.in_event_window and not prev.in_event_window:
        d = f" ({curr.days_to_event}d)" if curr.days_to_event is not None else ""
        return EngineEvent(
            code="event_window",
            severity="warn",
            detail=f"entering {curr.next_event or 'event'} risk window{d} — binary event ahead",
            at=curr.at,
        )
    return None


def _financial_conditions_tighten(
    prev: MarketState, curr: MarketState, cfg: EventConfig
) -> EngineEvent | None:
    if curr.financial_conditions is None or prev.financial_conditions is None:
        return None
    if prev.financial_conditions <= cfg.nfci_tight < curr.financial_conditions:
        return EngineEvent(
            code="financial_conditions_tighten",
            severity="warn",
            detail=(
                f"NFCI {prev.financial_conditions:+.2f} -> {curr.financial_conditions:+.2f} "
                "(financial conditions tightening)"
            ),
            at=curr.at,
        )
    return None


def _vix_backwardation(
    prev: MarketState, curr: MarketState, cfg: EventConfig
) -> EngineEvent | None:
    if curr.vix_term_structure is None or prev.vix_term_structure is None:
        return None
    if prev.vix_term_structure >= cfg.vix_backwardation > curr.vix_term_structure:
        return EngineEvent(
            code="vix_backwardation",
            severity="critical",
            detail=(
                f"VIX term structure inverted ({curr.vix_term_structure:.2f}) — "
                "acute near-term stress"
            ),
            at=curr.at,
        )
    return None


def _credit_stress(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    # Fires once when HY OAS crosses ABOVE the stress level (bond market pricing risk).
    if curr.hy_oas is None or prev.hy_oas is None:
        return None
    if prev.hy_oas < cfg.hy_oas_stress <= curr.hy_oas:
        sev = "critical" if curr.hy_oas >= cfg.hy_oas_high else "warn"
        return EngineEvent(
            code="credit_stress",
            severity=sev,
            detail=(
                f"HY OAS {prev.hy_oas:.1f}% -> {curr.hy_oas:.1f}% "
                "(credit spreads widening past stress)"
            ),
            at=curr.at,
        )
    return None


def _recession_onset(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    # Fires once when the macro recession-risk read crosses INTO "high".
    if curr.recession_risk_label != "high" or prev.recession_risk_label == "high":
        return None
    rr = f" ({curr.recession_risk:.2f})" if curr.recession_risk is not None else ""
    cyc = f", cycle={curr.macro_cycle_label}" if curr.macro_cycle_label else ""
    return EngineEvent(
        code="recession_onset",
        severity="critical",
        detail=f"macro recession-risk -> high{rr}{cyc}",
        at=curr.at,
    )


def _halt(prev: MarketState, curr: MarketState, cfg: EventConfig) -> EngineEvent | None:
    if curr.halt_active and not prev.halt_active:
        return EngineEvent(
            code="halt", severity="critical", detail="kill-switch HALT became active", at=curr.at
        )
    return None


def _drawdown(
    curr: MarketState, cfg: EventConfig, *, session_high_equity: float | None
) -> EngineEvent | None:
    if curr.equity is None or session_high_equity is None or session_high_equity <= 0:
        return None
    dd = curr.equity / session_high_equity - 1.0
    if dd > cfg.intraday_drawdown_pct:
        return None
    sev = "critical" if dd <= 2 * cfg.intraday_drawdown_pct else "warn"
    return EngineEvent(
        code="drawdown",
        severity=sev,
        detail=f"intraday drawdown {dd:.2%} from session high ${session_high_equity:,.0f}",
        at=curr.at,
    )


# State-to-state detectors (each needs a non-None prev to fire on a *transition*;
# the intraday + risk detectors are ABSOLUTE — they evaluate `curr` only).
_PAIR_DETECTORS = (
    _regime_flip,
    _posture_cross,
    _vol_spike,
    _breadth_collapse,
    _corr_spike,
    _risk_breach,
    _intraday_selloff,
    _intraday_breadth_break,
    _intraday_range_spike,
    _negative_news,
    _event_window,
    _financial_conditions_tighten,
    _vix_backwardation,
    _credit_stress,
    _recession_onset,
    _halt,
)


def detect_events(
    prev: MarketState | None,
    curr: MarketState,
    cfg: EventConfig | None = None,
    *,
    session_high_equity: float | None = None,
) -> list[EngineEvent]:
    """All material events from ``prev`` -> ``curr``. First cycle (prev=None)
    only evaluates absolute (non-transition) conditions, so the engine doesn't
    fire a burst of stale 'changes' on startup."""
    cfg = cfg or EventConfig()
    out: list[EngineEvent] = []
    base = prev if prev is not None else curr  # absolute checks compare to self -> no transition
    for det in _PAIR_DETECTORS:
        ev = det(base, curr, cfg)
        if ev is not None:
            out.append(ev)
    dd = _drawdown(curr, cfg, session_high_equity=session_high_equity)
    if dd is not None:
        out.append(dd)
    return out
