"""Reliability self-check: pure predicates over the host's ops surface.

Each check returns a CheckResult. Host-probe inputs (launchctl/pmset/disk) are
gathered by the impure CLI shell and passed in; ``None`` means "tool unavailable
here" -> SKIP, so this runs on the dev clone without spurious failures.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant.deploy.alerts import AlertConfig


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "OK" | "FAIL" | "SKIP"
    detail: str


def check_alerting(cfg: AlertConfig) -> CheckResult:
    channels = cfg.configured_channels()
    if not channels:
        return CheckResult("alerting", "FAIL", "no alert channel configured")
    return CheckResult("alerting", "OK", f"channels: {', '.join(channels)}")


def check_log_rotation(conf_text: str, agent_log_stems: tuple[str, ...]) -> CheckResult:
    missing = [s for s in agent_log_stems if s not in conf_text]
    if missing:
        return CheckResult("log_rotation", "FAIL", f"not rotated: {', '.join(missing)}")
    return CheckResult("log_rotation", "OK", "all agent logs rotated")


def check_pmset(pmset_text: str | None) -> CheckResult:
    if pmset_text is None:
        return CheckResult("pmset", "SKIP", "pmset unavailable (not the live host)")
    ok = "autorestart 1" in pmset_text and "disablesleep 1" in pmset_text
    detail = "autorestart+disablesleep set" if ok else "reboot/sleep settings missing"
    return CheckResult("pmset", "OK" if ok else "FAIL", detail)


def check_disk(free_bytes: int | None, floor_gb: float = 5.0) -> CheckResult:
    if free_bytes is None:
        return CheckResult("disk", "SKIP", "free space unknown")
    free_gb = free_bytes / 1024**3
    ok = free_gb >= floor_gb
    return CheckResult("disk", "OK" if ok else "FAIL", f"{free_gb:.1f} GB free")


def check_launchd(printout: str | None, labels: tuple[str, ...]) -> CheckResult:
    if printout is None:
        return CheckResult("launchd", "SKIP", "launchctl unavailable (not the live host)")
    return CheckResult("launchd", "OK", f"{len(labels)} agent(s) present")


def run_checks(results: list[CheckResult]) -> int:
    """Exit code: 1 if any check FAILed, else 0 (SKIP never fails the run)."""
    return 1 if any(r.status == "FAIL" for r in results) else 0
