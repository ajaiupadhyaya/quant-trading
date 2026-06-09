"""Manifest loader: parse + validate jobs.toml."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest

from quant.deploy.manifest import CatchUpPolicy, DayRule, load_manifest

REPO_MANIFEST = Path(__file__).resolve().parents[2] / "quant" / "deploy" / "jobs.toml"


def test_loads_all_jobs() -> None:
    m = load_manifest(REPO_MANIFEST)
    assert {j.name for j in m.jobs} == {
        "premarket-health",
        "daily-rebalance",
        "posttrade-reconciliation",
        "daily-digest",
        "daily-brief",
        "nightly-backtest",
        "regime-refresh",
        "weekly-grid-search",
        "weekly-validation-governance",
        "intraday-watch-open",
        "intraday-watch-midday",
        "intraday-watch-power-hour",
        "premarket-signals",
    }


def test_intraday_watch_jobs_are_readonly_and_page_safe() -> None:
    m = load_manifest(REPO_MANIFEST)
    watch = [j for j in m.jobs if j.name.startswith("intraday-watch")]
    assert len(watch) == 3
    for j in watch:
        # SAME_DAY (not NONE): a missed slot degrades to MISSED (a soft ping), never
        # the MISSED_CRITICAL emergency page the dispatcher fires for catch_up=NONE.
        assert j.catch_up == CatchUpPolicy.SAME_DAY
        assert j.timing_critical is False  # runs under the 'batch' lock, never trades
        assert j.commit_paths == ()  # the watch commits nothing
        assert j.commands == (
            ("analyst", "watch", "--slot", j.name.removeprefix("intraday-watch-")),
        )
    # the power-hour slot must close out before the 15:55 rebalance window
    ph = next(j for j in watch if j.name == "intraday-watch-power-hour")
    assert ph.max_lateness == time(15, 30)


def test_daily_rebalance_is_timing_critical_close_relative() -> None:
    m = load_manifest(REPO_MANIFEST)
    reb = next(j for j in m.jobs if j.name == "daily-rebalance")
    assert reb.timing_critical is True
    assert reb.catch_up == CatchUpPolicy.NONE
    assert reb.close_offset_min == 5
    assert reb.trigger_et is None
    assert reb.days == DayRule.WEEKDAYS_TRADING


def test_premarket_is_fixed_time_catchup_safe() -> None:
    m = load_manifest(REPO_MANIFEST)
    pre = next(j for j in m.jobs if j.name == "premarket-health")
    assert pre.trigger_et == time(9, 0)
    assert pre.catch_up == CatchUpPolicy.SAME_DAY
    assert pre.timing_critical is False


def test_commands_are_arg_tuples() -> None:
    m = load_manifest(REPO_MANIFEST)
    reb = next(j for j in m.jobs if j.name == "daily-rebalance")
    assert all(isinstance(c, tuple) for c in reb.commands)
    # The chain ends in a rebalance step. Tolerate the shakedown's "--dry-run"
    # flag so this stays green in both dry-run and live config.
    assert any(c[0] == "rebalance" for c in reb.commands)


def test_duplicate_names_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "dup.toml"
    bad.write_text(
        '[[job]]\nname="a"\ntrigger_et="09:00"\ndays="WEEKDAYS_TRADING"\n'
        'catch_up="SAME_DAY"\nmax_lateness="14:00"\nmax_runtime_s=600\n'
        'commands=[["doctor"]]\n'
        '[[job]]\nname="a"\ntrigger_et="10:00"\ndays="WEEKDAYS_TRADING"\n'
        'catch_up="SAME_DAY"\nmax_lateness="14:00"\nmax_runtime_s=600\ncommands=[["doctor"]]\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(bad)
