"""Pure due/catch-up/missed engine — the heart of the scheduler."""

from __future__ import annotations

from datetime import date, datetime, time

from quant.deploy.calendar_clock import ET
from quant.deploy.manifest import CatchUpPolicy, DayRule, Job, Manifest
from quant.deploy.scheduler import DispatchKind, due_jobs


def _et(y, mo, d, h, mi) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def _job(**kw) -> Job:
    base = dict(
        name="j",
        trigger_et=time(9, 0),
        close_offset_min=None,
        days=DayRule.WEEKDAYS_TRADING,
        catch_up=CatchUpPolicy.SAME_DAY,
        max_lateness=time(14, 0),
        max_lateness_next_day=False,
        max_runtime_s=600,
        timing_critical=False,
        commands=(("doctor",),),
        commit_paths=(),
    )
    base.update(kw)
    return Job(**base)


def _m(*jobs: Job) -> Manifest:
    return Manifest(jobs=tuple(jobs))


def test_fresh_fires_at_trigger() -> None:
    j = _job()
    out = due_jobs(_et(2026, 6, 2, 9, 1), _m(j), {})  # Tue, within 3-min window
    assert [d.kind for d in out] == [DispatchKind.FRESH]


def test_already_marked_today_not_due() -> None:
    j = _job()
    out = due_jobs(_et(2026, 6, 2, 9, 1), _m(j), {"j": date(2026, 6, 2)})
    assert out == []


def test_catch_up_past_window() -> None:
    j = _job()
    out = due_jobs(_et(2026, 6, 2, 11, 0), _m(j), {})  # past 9:03, before 14:00 horizon
    assert [d.kind for d in out] == [DispatchKind.CATCH_UP]


def test_missed_past_horizon() -> None:
    j = _job()
    out = due_jobs(_et(2026, 6, 2, 15, 0), _m(j), {})  # past 14:00 horizon
    assert [d.kind for d in out] == [DispatchKind.MISSED]


def test_holiday_no_fire() -> None:
    j = _job()
    # 2026-07-03 is the observed Independence Day holiday (July 4 is Saturday).
    assert due_jobs(_et(2026, 7, 3, 9, 1), _m(j), {}) == []


def test_timing_critical_fresh_window_before_close() -> None:
    j = _job(
        name="reb",
        trigger_et=None,
        close_offset_min=5,
        catch_up=CatchUpPolicy.NONE,
        timing_critical=True,
        max_lateness=time(16, 0),
    )
    out = due_jobs(_et(2026, 6, 2, 15, 56), _m(j), {})  # close 16:00 -> window 15:55-15:58
    assert [d.kind for d in out] == [DispatchKind.FRESH]


def test_timing_critical_missed_after_hard_cutoff() -> None:
    j = _job(
        name="reb",
        trigger_et=None,
        close_offset_min=5,
        catch_up=CatchUpPolicy.NONE,
        timing_critical=True,
        max_lateness=time(16, 0),
    )
    out = due_jobs(_et(2026, 6, 2, 16, 5), _m(j), {})  # past close-2min
    assert [d.kind for d in out] == [DispatchKind.MISSED_CRITICAL]


def test_timing_critical_early_close_day() -> None:
    j = _job(
        name="reb",
        trigger_et=None,
        close_offset_min=5,
        catch_up=CatchUpPolicy.NONE,
        timing_critical=True,
        max_lateness=time(16, 0),
    )
    # 2026-11-27 early close 13:00 -> rebalance window 12:55-12:58.
    assert [d.kind for d in due_jobs(_et(2026, 11, 27, 12, 56), _m(j), {})] == [DispatchKind.FRESH]
    # a 15:55 tick that day is past the 13:00 close -> MISSED_CRITICAL
    assert [d.kind for d in due_jobs(_et(2026, 11, 27, 15, 55), _m(j), {})] == [
        DispatchKind.MISSED_CRITICAL
    ]


def test_dst_summer_and_winter_same_et_wallclock() -> None:
    j = _job(trigger_et=time(9, 0))
    # both should be FRESH at 09:01 ET regardless of season
    assert due_jobs(_et(2026, 1, 15, 9, 1), _m(j), {})[0].kind == DispatchKind.FRESH  # EST
    assert due_jobs(_et(2026, 7, 15, 9, 1), _m(j), {})[0].kind == DispatchKind.FRESH  # EDT


def test_asleep_thursday_wake_friday_3am_does_not_prefire_friday() -> None:
    # Friday 03:00 ET: a WEEKDAYS_TRADING job triggering 09:00 is BEFORE its window.
    j = _job(trigger_et=time(9, 0))
    assert due_jobs(_et(2026, 6, 5, 3, 0), _m(j), {}) == []  # Fri pre-dawn -> not due


def test_evening_job_catch_up_after_midnight_attributes_to_prior_session() -> None:
    j = _job(
        name="nb",
        days=DayRule.TRADING_DAY_EVENING,
        trigger_et=time(22, 0),
        max_lateness=time(9, 0),
        max_lateness_next_day=True,
    )
    # Sat 01:30 ET, no marker -> Friday's (2026-06-05) backtest is caught up
    out = due_jobs(_et(2026, 6, 6, 1, 30), _m(j), {})
    assert len(out) == 1 and out[0].kind == DispatchKind.CATCH_UP
    assert out[0].session_date == date(2026, 6, 5)
