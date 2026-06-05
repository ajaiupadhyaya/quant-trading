"""Impure shell: one launchd tick. Reads the clock, loads manifest+markers,
calls the pure scheduler, and runs due jobs via injected collaborators.

Single-flight: a non-blocking per-job lock (fcntl.flock LOCK_NB) ensures two
overlapping ticks never run the same job twice; the timing-critical rebalance
has its OWN lock so a long batch job cannot block it.
"""

from __future__ import annotations

import fcntl
import subprocess
import time as _time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from quant.deploy.calendar_clock import to_et
from quant.deploy.manifest import Job, Manifest
from quant.deploy.markers import read_markers, write_marker
from quant.deploy.scheduler import Dispatch, DispatchKind, due_jobs
from quant.util.logging import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
UV_BIN = "/opt/homebrew/bin/uv"
STRATEGY_SLUGS = (
    "defensive-etf-allocation",
    "trend",
    "momentum",
    "risk-parity",
    "multi-factor",
    "pairs",
)


class Alerts(Protocol):
    def ping_success(self, url: str | None) -> None: ...
    def ping_fail(self, url: str | None, body: str = "") -> None: ...
    def send_emergency(self, title: str, message: str) -> bool: ...


Runner = Callable[[list[str], Path], int]


def _build_command(args: list[str]) -> list[str]:
    """Build the shell command for one expanded step.

    Normal steps run `uv run quant <args>` (the `quant` console entrypoint — a
    bare `uv run data refresh` would try to spawn an executable named `data`).
    The `__python__` sentinel runs a raw script: `uv run python <path...>`.
    """
    if args[:1] == ["__python__"]:
        return [UV_BIN, "run", "python", *args[1:]]
    return [UV_BIN, "run", "quant", *args]


def default_runner(args: list[str], cwd: Path) -> int:
    """Run an expanded step via `uv` and return its exit code."""
    return subprocess.run(_build_command(args), cwd=cwd, check=False).returncode


def _expand(job: Job) -> list[list[str]]:
    """Expand manifest command sentinels into concrete `quant`/python arg-lists."""
    out: list[list[str]] = []
    for cmd in job.commands:
        if cmd[0] == "__script__":
            out.append(["__python__", *cmd[1:]])
        elif cmd[0] == "__backtest_matrix__":
            flag = list(cmd[1:])
            for slug in STRATEGY_SLUGS:
                out.append(["backtest", slug, "--start", "2015-01-01", *flag])
        elif cmd[0] == "__validate_matrix__":
            for slug in STRATEGY_SLUGS:
                out.append(
                    [
                        "validate",
                        slug,
                        "--start",
                        "2010-01-01",
                        "--bootstrap-resamples",
                        "5000",
                        "--bootstrap-seed",
                        "0",
                    ]
                )
        else:
            out.append(list(cmd))
    return out


@contextmanager
def _job_lock(data_dir: Path, lock_name: str) -> Iterator[bool]:
    lock_dir = data_dir / "ops" / "scheduler"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{lock_name}.lock"
    fh = lock_path.open("w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


@dataclass
class Dispatcher:
    data_dir: Path
    manifest: Manifest
    runner: Runner = default_runner
    alerts: Alerts | None = None
    halt_active: Callable[[], bool] = lambda: False
    tick_url: str | None = None

    def tick(self, now_utc: datetime | None = None) -> int:
        now_utc = now_utc or datetime.now(UTC)
        now_et = to_et(now_utc)

        # Fresh-halt check first: a halted book never trades; alert immediately.
        if self.halt_active() and self.alerts is not None:
            self.alerts.send_emergency(
                "TRADING HALTED", f"halt active at {now_et:%Y-%m-%d %H:%M %Z}"
            )

        markers = read_markers(self.data_dir)
        dispatches = due_jobs(now_et, self.manifest, markers)
        any_failure = False
        for disp in dispatches:
            if disp.kind == DispatchKind.MISSED_CRITICAL:
                if self.alerts is not None:
                    self.alerts.send_emergency(
                        "REBALANCE WINDOW MISSED",
                        f"{disp.job.name} session {disp.session_date} not run before close cutoff",
                    )
                write_marker(
                    self.data_dir,
                    disp.job.name,
                    disp.session_date,
                    kind="MISSED_CRITICAL",
                    fired_at_utc=now_utc,
                    exit_code=-1,
                    duration_s=0.0,
                )
                continue
            if disp.kind == DispatchKind.MISSED:
                if self.alerts is not None:
                    self.alerts.ping_fail(
                        self.tick_url, f"{disp.job.name} MISSED {disp.session_date}"
                    )
                write_marker(
                    self.data_dir,
                    disp.job.name,
                    disp.session_date,
                    kind="MISSED",
                    fired_at_utc=now_utc,
                    exit_code=-1,
                    duration_s=0.0,
                )
                continue
            # FRESH or CATCH_UP -> run, under the job's lock (rebalance has its own).
            if self.halt_active() and disp.job.timing_critical:
                continue  # never trade while halted
            lock_name = "daily-rebalance" if disp.job.timing_critical else "batch"
            with _job_lock(self.data_dir, lock_name) as got:
                if not got:
                    logger.warning("lock {} busy; skipping {} this tick", lock_name, disp.job.name)
                    if disp.job.timing_critical and self.alerts is not None:
                        self.alerts.send_emergency(
                            "REBALANCE BLOCKED", f"{lock_name} lock held at trade window"
                        )
                    continue
                rc = self._run_chain(disp)
                if rc == 0:
                    write_marker(
                        self.data_dir,
                        disp.job.name,
                        disp.session_date,
                        kind=disp.kind,
                        fired_at_utc=now_utc,
                        exit_code=0,
                        duration_s=0.0,
                    )
                else:
                    any_failure = True
                    if self.alerts is not None:
                        self.alerts.ping_fail(self.tick_url, f"{disp.job.name} exit {rc}")

        # Suppress the success heartbeat if any job failed this tick.
        if not any_failure and self.alerts is not None:
            self.alerts.ping_success(self.tick_url)
        return 0

    def _run_chain(self, disp: Dispatch) -> int:
        start = _time.monotonic()
        for args in _expand(disp.job):
            rc = self.runner(args, REPO_ROOT)
            if rc != 0:
                logger.error("job {} step {} failed rc={}", disp.job.name, args, rc)
                return rc
        logger.info("job {} ok in {:.1f}s", disp.job.name, _time.monotonic() - start)
        return 0
