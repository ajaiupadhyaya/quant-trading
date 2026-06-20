"""Reliability self-check: pure predicates over the host's ops surface."""

from __future__ import annotations

from quant.deploy.alerts import AlertConfig
from quant.deploy.selfcheck import (
    CheckResult,
    check_alerting,
    check_disk,
    check_launchd,
    check_log_rotation,
    check_pmset,
    run_checks,
)


def test_alerting_fail_when_unconfigured() -> None:
    assert check_alerting(AlertConfig(None, None, None, None, None)).status == "FAIL"


def test_alerting_ok_when_configured() -> None:
    cfg = AlertConfig("https://hc", None, "t", "u", None)
    assert check_alerting(cfg).status == "OK"


def test_log_rotation_requires_all_stems() -> None:
    conf = "engine.stdout.log\ntick.stdout.log\n"
    r = check_log_rotation(conf, ("engine", "tick", "guard"))
    assert r.status == "FAIL" and "guard" in r.detail


def test_log_rotation_ok_when_all_present() -> None:
    conf = "engine.x\ntick.x\nguard.x\n"
    assert check_log_rotation(conf, ("engine", "tick", "guard")).status == "OK"


def test_pmset_ok_needs_autorestart_and_disablesleep() -> None:
    assert check_pmset(" autorestart 1\n disablesleep 1\n").status == "OK"
    assert check_pmset(" autorestart 0\n disablesleep 1\n").status == "FAIL"
    assert check_pmset(None).status == "SKIP"


def test_disk_floor() -> None:
    assert check_disk(10 * 1024**3).status == "OK"
    assert check_disk(1 * 1024**3).status == "FAIL"
    assert check_disk(None).status == "SKIP"


def test_launchd_present_or_skip() -> None:
    assert check_launchd("state = running\n", ("com.x",)).status == "OK"
    assert check_launchd(None, ("com.x",)).status == "SKIP"


def test_run_checks_returns_1_on_any_fail() -> None:
    assert run_checks([CheckResult("a", "OK", ""), CheckResult("b", "FAIL", "x")]) == 1
    assert run_checks([CheckResult("a", "OK", ""), CheckResult("c", "SKIP", "")]) == 0
