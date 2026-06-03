"""Scheduled event calendar + policy/macro risk read (Phase 7C).

The calendar is mostly COMPUTED (jobs report = first Friday, OpEx/quad-witching =
third Friday, US federal elections, quarter-end) so it is always correct; FOMC
announcement dates can't be computed and are embedded below — VERIFY/UPDATE them
annually from federalreserve.gov. The risk read layers FRED policy-uncertainty
(EPU), financial conditions (NFCI), stress (STLFSI4), and the VIX term structure
on top of event proximity. Everything is bounded + fail-open; advisory only.
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from quant.util.logging import logger

# FOMC announcement days (2nd day of each meeting). EMBEDDED — verify annually.
_FOMC_DATES: tuple[date, ...] = (
    date(2025, 12, 10),
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
    date(2027, 1, 27),
    date(2027, 3, 17),
)

_FRIDAY = 4
_TUESDAY = 1
_MONDAY = 0


@dataclass(frozen=True)
class ScheduledEvent:
    name: str
    date: str  # ISO
    impact: str  # "high" | "medium"


@dataclass(frozen=True)
class EventRiskConfig:
    event_window_days: int = 2  # within this many days of a HIGH-impact event
    horizon_days: int = 21
    epu_elevated: float = 200.0  # US EPU above this = elevated policy uncertainty
    nfci_tight: float = 0.0  # NFCI > 0 = tighter-than-average financial conditions
    finstress_high: float = 1.0
    vix_backwardation: float = 1.0  # VXV/VIX below this = near-term stress (backwardated)


@dataclass(frozen=True)
class EventRisk:
    next_event: str | None
    next_event_date: str | None
    days_to_event: int | None
    in_event_window: bool
    policy_uncertainty: float | None  # EPU level
    policy_uncertainty_elevated: bool
    financial_conditions: float | None  # NFCI
    financial_stress: float | None  # STLFSI4
    vix_term_structure: float | None  # VXV/VIX (>1 contango/calm, <1 backwardation/stress)
    risk_label: str | None  # "calm" | "watch" | "stressed"


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_business_day(year: int, month: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() >= 5:  # back up over Sat/Sun
        d -= timedelta(days=1)
    return d


def _nfp(year: int, month: int) -> date:
    """Nonfarm payrolls release ≈ first Friday of the month."""
    return _nth_weekday(year, month, _FRIDAY, 1)


def _opex(year: int, month: int) -> date:
    """Monthly options expiration = third Friday."""
    return _nth_weekday(year, month, _FRIDAY, 3)


def _election_day(year: int) -> date | None:
    """US federal election day (even years): first Tuesday after the first Monday in Nov."""
    if year % 2 != 0:
        return None
    first_monday = _nth_weekday(year, 11, _MONDAY, 1)
    return first_monday + timedelta(days=1)  # the Tuesday after the first Monday


def _months_ahead(asof: date, n: int) -> tuple[int, int]:
    m = asof.month - 1 + n
    return asof.year + m // 12, m % 12 + 1


def upcoming_events(asof: date, *, horizon_days: int = 21) -> list[ScheduledEvent]:
    """High/medium-impact scheduled events in ``[asof, asof+horizon_days]``, sorted."""
    end = asof + timedelta(days=horizon_days)
    cand: list[ScheduledEvent] = []

    for d in _FOMC_DATES:
        cand.append(ScheduledEvent("FOMC", d.isoformat(), "high"))

    # Recurring monthly events for this + the next two months (covers the horizon).
    for k in range(0, 3):
        y, m = _months_ahead(asof, k)
        cand.append(ScheduledEvent("Jobs report", _nfp(y, m).isoformat(), "high"))
        opex = _opex(y, m)
        quad = m in (3, 6, 9, 12)
        cand.append(
            ScheduledEvent("Quad-witching" if quad else "OpEx", opex.isoformat(),
                           "high" if quad else "medium")
        )
        if m in (3, 6, 9, 12):
            cand.append(ScheduledEvent("Quarter-end", _last_business_day(y, m).isoformat(), "medium"))

    for yr in {asof.year, end.year}:
        e = _election_day(yr)
        if e is not None:
            cand.append(ScheduledEvent("US election", e.isoformat(), "high"))

    out = [ev for ev in cand if asof <= date.fromisoformat(ev.date) <= end]
    return sorted(out, key=lambda ev: (ev.date, ev.impact))


def next_high_impact_event(asof: date) -> ScheduledEvent | None:
    """Soonest HIGH-impact scheduled event on/after ``asof`` (looks out ~120 days)."""
    events = upcoming_events(asof, horizon_days=120)
    for ev in events:
        if ev.impact == "high":
            return ev
    return None


def _finite(x: Any) -> float | None:
    try:
        v = float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and math.isfinite(v)) else None


def compute_event_risk(
    asof: date,
    *,
    epu: float | None = None,
    nfci: float | None = None,
    finstress: float | None = None,
    vix: float | None = None,
    vix3m: float | None = None,
    config: EventRiskConfig | None = None,
) -> EventRisk:
    """Pure: combine event proximity + macro-risk inputs into one read."""
    cfg = config or EventRiskConfig()
    nxt = next_high_impact_event(asof)
    days_to = (date.fromisoformat(nxt.date) - asof).days if nxt is not None else None
    in_window = days_to is not None and 0 <= days_to <= cfg.event_window_days

    epu_v = _finite(epu)
    nfci_v = _finite(nfci)
    fs_v = _finite(finstress)
    vix_v = _finite(vix)
    vix3m_v = _finite(vix3m)
    term = (vix3m_v / vix_v) if (vix_v is not None and vix3m_v is not None and vix_v > 0) else None
    epu_elevated = epu_v is not None and epu_v > cfg.epu_elevated

    stressed = (
        (term is not None and term < cfg.vix_backwardation)
        or (nfci_v is not None and nfci_v > 0.5)
        or (fs_v is not None and fs_v > cfg.finstress_high)
    )
    watch = (
        in_window
        or epu_elevated
        or (nfci_v is not None and nfci_v > cfg.nfci_tight)
        or (term is not None and term < 1.05)
    )
    risk_label = "stressed" if stressed else ("watch" if watch else "calm")

    return EventRisk(
        next_event=(nxt.name if nxt else None),
        next_event_date=(nxt.date if nxt else None),
        days_to_event=days_to,
        in_event_window=in_window,
        policy_uncertainty=epu_v,
        policy_uncertainty_elevated=epu_elevated,
        financial_conditions=nfci_v,
        financial_stress=fs_v,
        vix_term_structure=_finite(term),
        risk_label=risk_label,
    )


def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def live_event_risk(
    settings: Any, asof: date, *, config: EventRiskConfig | None = None
) -> EventRisk:
    """Bounded, fail-open FRED reads + calendar. Never raises."""

    def _series_last(key: str) -> float | None:
        try:
            from quant.data import macro

            code = macro.FRED_SERIES.get(key, key)
            s = _with_timeout(lambda: macro.get_series(code), 8.0)
            ser = s.dropna() if s is not None else None
            return _finite(ser.iloc[-1]) if ser is not None and len(ser) else None
        except Exception as exc:  # one series failing must not sink the read
            logger.info("macro.events: series {} skipped ({!r})", key, exc)
            return None

    return compute_event_risk(
        asof,
        epu=_series_last("epu"),
        nfci=_series_last("nfci"),
        finstress=_series_last("finstress"),
        vix=_series_last("vix"),
        vix3m=_series_last("vix3m"),
        config=config,
    )


def render_event_risk(r: EventRisk | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if r is None:
        return "Event risk: unavailable"
    bits: list[str] = []
    if r.risk_label:
        bits.append(f"macro-risk={r.risk_label}")
    if r.next_event and r.days_to_event is not None:
        win = " [WINDOW]" if r.in_event_window else ""
        bits.append(f"next={r.next_event} in {r.days_to_event}d{win}")
    if r.policy_uncertainty is not None:
        flag = "!" if r.policy_uncertainty_elevated else ""
        bits.append(f"EPU={r.policy_uncertainty:.0f}{flag}")
    if r.financial_conditions is not None:
        bits.append(f"NFCI={r.financial_conditions:+.2f}")
    if r.vix_term_structure is not None:
        bits.append(f"vix_term={r.vix_term_structure:.2f}")
    return "Event risk: " + (", ".join(bits) if bits else "n/a")
