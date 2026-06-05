"""The manifest must faithfully carry the migrated jobs + the timing-critical
pairing invariant the scheduler relies on."""

from __future__ import annotations

from pathlib import Path

from quant.deploy.manifest import CatchUpPolicy, load_manifest

MANIFEST = Path(__file__).resolve().parents[2] / "quant" / "deploy" / "jobs.toml"


def test_timing_critical_iff_catchup_none() -> None:
    for j in load_manifest(MANIFEST).jobs:
        assert j.timing_critical == (j.catch_up == CatchUpPolicy.NONE), j.name


def test_only_daily_rebalance_is_timing_critical() -> None:
    crit = [j.name for j in load_manifest(MANIFEST).jobs if j.timing_critical]
    assert crit == ["daily-rebalance"]


def test_committing_jobs_declare_commit_paths() -> None:
    m = load_manifest(MANIFEST)
    by = {j.name: j for j in m.jobs}
    assert "docs/live-recon/" in by["posttrade-reconciliation"].commit_paths
    assert "data/governance/" in by["weekly-validation-governance"].commit_paths


def test_retired_workflows_have_schedule_disabled() -> None:
    wf = Path(__file__).resolve().parents[2] / ".github" / "workflows"
    for name in (
        "daily-rebalance",
        "premarket-health",
        "posttrade-reconciliation",
        "nightly-backtest",
        "weekly-grid-search",
        "weekly-validation-governance",
    ):
        text = (wf / f"{name}.yml").read_text()
        assert "SCHEDULE RETIRED" in text, name
        # the active `schedule:` trigger is gone (only commented references remain)
        active = [ln for ln in text.splitlines() if ln.strip().startswith("schedule:")]
        assert active == [], f"{name} still has an active schedule: trigger"
        assert "workflow_dispatch" in text, name
