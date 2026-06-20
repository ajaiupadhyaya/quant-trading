"""Per-job runtime budget: a hung job is killed (124) and the chain aborts."""

from __future__ import annotations

import subprocess
from datetime import date, time
from pathlib import Path

import pytest

from quant.deploy import dispatcher as dispatcher_mod
from quant.deploy.dispatcher import Dispatcher, default_runner
from quant.deploy.manifest import CatchUpPolicy, DayRule, Job, Manifest
from quant.deploy.scheduler import Dispatch, DispatchKind


def _job(max_runtime_s: int) -> Job:
    return Job(
        name="slow",
        trigger_et=time(10, 0),
        close_offset_min=None,
        days=DayRule.WEEKDAYS_TRADING,
        catch_up=CatchUpPolicy.SAME_DAY,
        max_lateness=time(1, 0),
        max_lateness_next_day=False,
        max_runtime_s=max_runtime_s,
        timing_critical=False,
        commands=(("noop",),),
        commit_paths=(),
    )


def _dispatch(job: Job) -> Dispatch:
    return Dispatch(job=job, kind=DispatchKind.FRESH, session_date=date(2026, 6, 22))


def test_default_runner_returns_124_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["noop"], timeout=0.1)

    monkeypatch.setattr(dispatcher_mod.subprocess, "run", fake_run)
    assert default_runner(["noop"], tmp_path, 0.1) == 124


def test_run_chain_aborts_when_budget_exhausted(tmp_path: Path) -> None:
    called: list[list[str]] = []

    def runner(args: list[str], cwd: Path, timeout_s: float | None) -> int:
        called.append(args)
        return 0

    # max_runtime_s=0 -> deadline is now -> the chain aborts before the first step.
    disp = Dispatcher(data_dir=tmp_path, manifest=Manifest(jobs=()), runner=runner)
    assert disp._run_chain(_dispatch(_job(0))) == 124
    assert called == []  # never ran a step past the deadline


def test_run_chain_passes_timeout_budget_to_runner(tmp_path: Path) -> None:
    seen: list[float | None] = []

    def ok_runner(args: list[str], cwd: Path, timeout_s: float | None) -> int:
        seen.append(timeout_s)
        return 0

    disp = Dispatcher(data_dir=tmp_path, manifest=Manifest(jobs=()), runner=ok_runner)
    assert disp._run_chain(_dispatch(_job(30))) == 0
    assert seen and seen[0] is not None and 0 < seen[0] <= 30.0
