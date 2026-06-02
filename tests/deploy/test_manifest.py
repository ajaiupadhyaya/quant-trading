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
        "nightly-backtest",
        "weekly-grid-search",
        "weekly-validation-governance",
    }


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
