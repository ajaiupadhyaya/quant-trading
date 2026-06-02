"""Pure scheduler: decide which manifest jobs are due this tick and as what kind.

Two disjoint ladders so a job is never in two classes:
  * catch-up-safe (catch_up=SAME_DAY):  FRESH -> CATCH_UP -> MISSED
  * timing-critical (catch_up=NONE):    FRESH -> MISSED_CRITICAL  (no catch-up)

Session attribution (`_session_date`) maps the tick to the trading session a job
serves; for TRADING_DAY_EVENING the session can be the prior calendar day when
ticking in the 00:00-09:00 catch-up window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from quant.deploy.calendar_clock import ET, session_close_et
from quant.deploy.manifest import DayRule, Job, Manifest
from quant.util.trading_calendar import is_trading_day, previous_trading_day

FRESH_TOL_MIN = 3
CRITICAL_CUTOFF_BEFORE_CLOSE_MIN = 2


class DispatchKind(StrEnum):
    FRESH = "FRESH"
    CATCH_UP = "CATCH_UP"
    MISSED = "MISSED"
    MISSED_CRITICAL = "MISSED_CRITICAL"


@dataclass(frozen=True)
class Dispatch:
    job: Job
    kind: DispatchKind
    session_date: date


def _session_date(job: Job, now_et: datetime) -> date | None:
    d = now_et.date()
    t = now_et.time()
    if job.days == DayRule.WEEKDAYS_TRADING:
        return d if is_trading_day(d) else None
    if job.days == DayRule.SATURDAY:
        return d if d.weekday() == 5 else None
    if job.days == DayRule.TRADING_DAY_EVENING:
        if t >= time(22, 0):
            return d if is_trading_day(d) else None
        if t <= time(9, 0):
            return previous_trading_day(d)  # always a trading day
        return None
    return None  # type: ignore[unreachable]  # exhaustive over DayRule StrEnum


def _trigger_time(job: Job, d: date) -> time:
    if job.close_offset_min is not None:
        close = datetime.combine(d, session_close_et(d))
        return (close - timedelta(minutes=job.close_offset_min)).time()
    assert job.trigger_et is not None
    return job.trigger_et


def _combine(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET)


def _classify(job: Job, now_et: datetime, d: date) -> DispatchKind | None:
    trigger = _trigger_time(job, d)
    # For the evening job the trigger datetime is on the SESSION date d (22:00),
    # while now_et may be d+1 in the morning. Combine on d.
    trig_dt = _combine(d, trigger)

    if job.timing_critical:
        close_dt = _combine(d, session_close_et(d))
        hard_cutoff = close_dt - timedelta(minutes=CRITICAL_CUTOFF_BEFORE_CLOSE_MIN)
        if now_et < trig_dt:
            return None
        if trig_dt <= now_et <= hard_cutoff:
            return DispatchKind.FRESH
        return DispatchKind.MISSED_CRITICAL

    fresh_end = trig_dt + timedelta(minutes=FRESH_TOL_MIN)
    horizon_day = d + timedelta(days=1) if job.max_lateness_next_day else d
    horizon_end = _combine(horizon_day, job.max_lateness)
    if now_et < trig_dt:
        return None
    if trig_dt <= now_et <= fresh_end:
        return DispatchKind.FRESH
    if fresh_end < now_et <= horizon_end:
        return DispatchKind.CATCH_UP
    return DispatchKind.MISSED


def due_jobs(now_et: datetime, manifest: Manifest, markers: Mapping[str, date]) -> list[Dispatch]:
    out: list[Dispatch] = []
    for job in manifest.jobs:
        d = _session_date(job, now_et)
        if d is None:
            continue
        if markers.get(job.name) == d:
            continue  # already handled this session (any kind)
        kind = _classify(job, now_et, d)
        if kind is None:
            continue
        out.append(Dispatch(job=job, kind=kind, session_date=d))
    return out
