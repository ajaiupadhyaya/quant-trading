"""The premarket-signals job is a read-only, page-safe, pre-open quant job."""

from __future__ import annotations

import subprocess
from datetime import time
from pathlib import Path

from quant.deploy.manifest import CatchUpPolicy, DayRule, load_manifest

REPO = Path(__file__).resolve().parents[2]
REPO_MANIFEST = REPO / "quant" / "deploy" / "jobs.toml"


def _job():
    m = load_manifest(REPO_MANIFEST)
    return next(j for j in m.jobs if j.name == "premarket-signals")


def test_premarket_signals_present() -> None:
    assert _job().name == "premarket-signals"


def test_is_readonly_and_page_safe() -> None:
    j = _job()
    # SAME_DAY (not NONE): a missed slot is a soft ping, never the MISSED_CRITICAL page.
    assert j.catch_up == CatchUpPolicy.SAME_DAY
    assert j.timing_critical is False  # shares the batch lock, never the rebalance path
    assert j.commit_paths == ()  # host-local log only
    assert j.max_lateness_next_day is False


def test_fixed_time_after_premarket_health() -> None:
    m = load_manifest(REPO_MANIFEST)
    sig = next(j for j in m.jobs if j.name == "premarket-signals")
    health = next(j for j in m.jobs if j.name == "premarket-health")
    assert sig.close_offset_min is None
    assert sig.trigger_et == time(9, 15)
    assert sig.days == DayRule.WEEKDAYS_TRADING
    assert health.trigger_et is not None and sig.trigger_et > health.trigger_et


def test_command_is_research_signals() -> None:
    assert _job().commands == (("research", "signals"),)


def test_max_runtime_bounded() -> None:
    assert _job().max_runtime_s == 120


def test_signals_logs_are_gitignored() -> None:
    for rel in ("data/research/signals.jsonl", "data/research/signals_latest.json"):
        out = subprocess.run(["git", "check-ignore", rel], cwd=REPO, capture_output=True, text=True)
        assert out.returncode == 0, f"{rel} is not gitignored"
