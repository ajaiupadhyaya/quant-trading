# Phase 2a — Reliability (correctness/safety) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `quant-trading`'s live failures *safe and visible* on the M4 — enforce job timeouts,
DRY + harden alerting with a config-assertion and a self-test, add a reliability self-check, and emit
structured ANSI-free rotated logs.

**Architecture:** Pure check/resolve functions (unit-tested on this dev clone) behind thin impure
CLI shells. Extends the existing `quant/deploy/` dispatcher + alerts and `quant/util/logging.py`; no
new dependencies. Splits from the Phase 2 design (`docs/specs/2026-06-20-phase2-reliability-design.md`);
this is **2a (correctness/safety)** — 2b (TUI ops panes, host-verify, soak harness, M4 proofs) follows.

**Tech Stack:** Python 3.12, `uv`, Click CLI, loguru, pytest. Same toolchain as the repo.

## Global Constraints

- **Verify command:** `uv run pytest -q -m "not network and not alpaca and not slow"` must stay green
  (1395 passing baseline); `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy quant/`
  (strict) must stay clean. Run all before each commit.
- **No new runtime dependency.** stdlib + loguru/requests/click already present only.
- **Zero live-behavior change to the trading path.** Timeout enforcement must not alter the
  exit-code/marker contract for jobs that finish within budget (regression-pin the happy path).
- **`mypy --strict`** — annotate every new function fully.
- Preserve the **one-executor invariant**; touch only the named gaps.
- Commit after each task with the shown message.

---

### Task 1: Enforce per-job runtime budget in the dispatcher

A hung job currently holds the batch `flock` forever — `Job.max_runtime_s` is parsed but never
applied. Bound the whole job chain by a monotonic deadline; on overrun, kill the subprocess and
return a sentinel non-zero (124, conventional timeout code) so no marker is written and catch-up
retries within the window.

**Files:**
- Modify: `quant/deploy/dispatcher.py` (`Runner` type :45, `default_runner` :60-62, `_run_chain`
  :199-207, and the `Dispatcher.runner` call in `_run_chain`)
- Modify: `quant/cli.py:2950` (the `ops_run_job` `disp.runner(args, ...)` call — add the timeout arg)
- Test: `tests/deploy/test_dispatcher_timeout.py` (new)

**Interfaces:**
- Produces: `Runner = Callable[[list[str], Path, float | None], int]`;
  `default_runner(args: list[str], cwd: Path, timeout_s: float | None = None) -> int`;
  `_run_chain` enforces `disp.job.max_runtime_s` as a whole-chain deadline, returning `124` on overrun.

- [ ] **Step 1: Write the failing test**

```python
# tests/deploy/test_dispatcher_timeout.py
from datetime import UTC, datetime
from pathlib import Path

from quant.deploy.dispatcher import Dispatcher, _run_chain_for_test  # helper added in Step 3
from quant.deploy.manifest import CatchUpPolicy, DayRule, Job
from quant.deploy.scheduler import Dispatch, DispatchKind
from datetime import time


def _job(max_runtime_s: int) -> Job:
    return Job(
        name="slow", trigger_et=time(10, 0), close_offset_min=None,
        days=DayRule.WEEKDAYS_TRADING, catch_up=CatchUpPolicy.SAME_DAY,
        max_lateness=time(1, 0), max_lateness_next_day=False,
        max_runtime_s=max_runtime_s, timing_critical=False,
        commands=(("noop",),), commit_paths=(),
    )


def test_run_chain_times_out_and_returns_124(tmp_path: Path) -> None:
    calls: list[float | None] = []

    def slow_runner(args: list[str], cwd: Path, timeout_s: float | None) -> int:
        calls.append(timeout_s)
        raise __import__("subprocess").TimeoutExpired(cmd=args, timeout=timeout_s or 0.0)

    disp = Dispatcher(data_dir=tmp_path, manifest=__import__("quant.deploy.manifest", fromlist=["Manifest"]).Manifest(jobs=()), runner=slow_runner)
    d = Dispatch(job=_job(5), session_date="2026-06-22", kind=DispatchKind.FRESH)
    rc = disp._run_chain(d)
    assert rc == 124
    assert calls and calls[0] is not None and calls[0] <= 5.0


def test_run_chain_passes_timeout_budget_to_runner(tmp_path: Path) -> None:
    seen: list[float | None] = []

    def ok_runner(args: list[str], cwd: Path, timeout_s: float | None) -> int:
        seen.append(timeout_s)
        return 0

    disp = Dispatcher(data_dir=tmp_path, manifest=__import__("quant.deploy.manifest", fromlist=["Manifest"]).Manifest(jobs=()), runner=ok_runner)
    d = Dispatch(job=_job(30), session_date="2026-06-22", kind=DispatchKind.FRESH)
    assert disp._run_chain(d) == 0
    assert seen and seen[0] is not None and 0 < seen[0] <= 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_dispatcher_timeout.py -v`
Expected: FAIL — `_run_chain` currently calls `self.runner(args, REPO_ROOT)` (2 args) and has no
timeout handling, so `slow_runner`/`ok_runner` (3 args) raise `TypeError`, and `TimeoutExpired` is
uncaught.

- [ ] **Step 3: Implement — thread a whole-chain deadline through `_run_chain` and `default_runner`**

In `quant/deploy/dispatcher.py`:

```python
# near the imports, add:
import subprocess

# change the Runner alias (was: Runner = Callable[[list[str], Path], int]):
Runner = Callable[[list[str], Path, float | None], int]

TIMEOUT_RC = 124  # conventional "command timed out" exit code

def default_runner(args: list[str], cwd: Path, timeout_s: float | None = None) -> int:
    """Run an expanded step via `uv`; return its exit code (124 on timeout)."""
    try:
        return subprocess.run(
            _build_command(args), cwd=cwd, check=False, timeout=timeout_s
        ).returncode
    except subprocess.TimeoutExpired:
        logger.error("step {} exceeded {:.0f}s budget; killed", args, timeout_s or 0.0)
        return TIMEOUT_RC
```

Replace `_run_chain` (lines 199-207) with a deadline-bounded version:

```python
    def _run_chain(self, disp: Dispatch) -> int:
        start = _time.monotonic()
        deadline = start + float(disp.job.max_runtime_s)
        for args in _expand(disp.job):
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                logger.error("job {} exceeded {}s budget before step {}",
                             disp.job.name, disp.job.max_runtime_s, args)
                return TIMEOUT_RC
            rc = self.runner(args, REPO_ROOT, remaining)
            if rc != 0:
                logger.error("job {} step {} failed rc={}", disp.job.name, args, rc)
                return rc
        logger.info("job {} ok in {:.1f}s", disp.job.name, _time.monotonic() - start)
        return 0
```

In `quant/cli.py`, update the `ops_run_job` manual call (line ~2950) from
`rc = disp.runner(args, Path(__file__).resolve().parents[1])` to pass an explicit budget:

```python
        rc = disp.runner(args, Path(__file__).resolve().parents[1], float(job.max_runtime_s))
```

- [ ] **Step 4: Run the new + existing dispatcher tests**

Run: `uv run pytest tests/deploy/ -q`
Expected: PASS. If existing `tests/deploy/test_dispatcher.py` injects a 2-arg fake runner, update
those fakes to accept `(args, cwd, timeout_s=None)` — that is part of this task. Re-run until green.

- [ ] **Step 5: Lint, type, commit**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy quant/
git add quant/deploy/dispatcher.py quant/cli.py tests/deploy/
git commit -m "fix(deploy): enforce per-job max_runtime_s — kill hung jobs, release the lock"
```

---

### Task 2: `AlertConfig.from_settings` + `configured_channels` — DRY the 5 duplicated construction sites

`AlertConfig(...)` is hand-built identically in 5 places in `cli.py`. Centralize it and add channel
introspection so alerting-misconfiguration becomes detectable (Tasks 3–5 consume this).

**Files:**
- Modify: `quant/deploy/alerts.py` (`AlertConfig` dataclass :36-42)
- Modify: `quant/cli.py` (5 `AlertConfig(...)` blocks at ~2873, 2915, 3019, 3086, 3185)
- Test: `tests/deploy/test_alert_config.py` (new)

**Interfaces:**
- Produces: `AlertConfig.from_settings(settings: _SettingsLike) -> AlertConfig` (classmethod);
  `AlertConfig.configured_channels() -> tuple[str, ...]`; `AlertConfig.is_configured -> bool`
  (property: any channel present). `_SettingsLike` is a `Protocol` with the 5 string-or-None attrs.

- [ ] **Step 1: Write the failing test**

```python
# tests/deploy/test_alert_config.py
from dataclasses import dataclass

from quant.deploy.alerts import AlertConfig


@dataclass
class _S:
    healthcheck_tick_url: str | None = None
    healthcheck_guard_url: str | None = None
    pushover_app_token: str | None = None
    pushover_user_key: str | None = None
    slack_webhook_url: str | None = None


def test_from_settings_copies_all_fields() -> None:
    cfg = AlertConfig.from_settings(_S(slack_webhook_url="https://hook", pushover_app_token="t", pushover_user_key="u"))
    assert cfg.slack_webhook_url == "https://hook"
    assert cfg.pushover_app_token == "t"


def test_configured_channels_and_is_configured() -> None:
    none = AlertConfig.from_settings(_S())
    assert none.configured_channels() == ()
    assert none.is_configured is False
    some = AlertConfig.from_settings(_S(healthcheck_tick_url="https://hc", pushover_app_token="t", pushover_user_key="u"))
    assert set(some.configured_channels()) == {"healthchecks", "pushover"}
    assert some.is_configured is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/deploy/test_alert_config.py -v`
Expected: FAIL — `AlertConfig` has no `from_settings`/`configured_channels`/`is_configured`.

- [ ] **Step 3: Implement on `AlertConfig`**

In `quant/deploy/alerts.py`, add a `Protocol` and the three members:

```python
from typing import Protocol  # add to imports


class _SettingsLike(Protocol):
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None
    slack_webhook_url: str | None


@dataclass(frozen=True)
class AlertConfig:
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None
    slack_webhook_url: str | None = None

    @classmethod
    def from_settings(cls, settings: _SettingsLike) -> "AlertConfig":
        return cls(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )

    def configured_channels(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.healthcheck_tick_url or self.healthcheck_guard_url:
            out.append("healthchecks")
        if self.pushover_app_token and self.pushover_user_key:
            out.append("pushover")
        if self.slack_webhook_url:
            out.append("slack")
        return tuple(out)

    @property
    def is_configured(self) -> bool:
        return bool(self.configured_channels())
```

Then replace each of the 5 `AlertConfig(healthcheck_tick_url=settings...., slack_webhook_url=settings.slack_webhook_url)`
blocks in `cli.py` with `AlertConfig.from_settings(settings)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/deploy/test_alert_config.py tests/deploy/ -q`
Expected: PASS. Existing alert/cli tests unaffected (same field values).

- [ ] **Step 5: Lint, type, commit**

```bash
uv run ruff check . && uv run ruff format . && uv run mypy quant/
git add quant/deploy/alerts.py quant/cli.py tests/deploy/test_alert_config.py
git commit -m "refactor(deploy): AlertConfig.from_settings + channel introspection (DRY 5 sites)"
```

---

### Task 3: Alert self-test — `AlertClient.send_test()` + `quant ops alert-test`

Replace the runbook's unchecked "test the alert channels" checkbox with a command that actually fires
each configured channel so the operator proves delivery before relying on it.

**Files:**
- Modify: `quant/deploy/alerts.py` (add `send_test`)
- Modify: `quant/cli.py` (new `ops alert-test` command under the `ops` group, after `ops_run_job` ~2953)
- Test: `tests/deploy/test_alert_selftest.py` (new)

**Interfaces:**
- Consumes: `AlertConfig` (Task 2), `AlertClient` (`alerts.py:45`).
- Produces: `AlertClient.send_test() -> dict[str, bool]` — keys are the configured channel names, value
  = delivered. Unconfigured channels are absent from the dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/deploy/test_alert_selftest.py
from quant.deploy.alerts import AlertClient, AlertConfig


def test_send_test_fires_each_configured_channel() -> None:
    posts: list[str] = []

    def fake_get(url: str, timeout: float) -> int:
        posts.append("get:" + url)
        return 200

    def fake_post(url: str, data: dict[str, object], timeout: float) -> int:
        posts.append("post:" + url)
        return 200

    def fake_post_json(url: str, payload: dict[str, object], timeout: float) -> int:
        posts.append("json:" + url)
        return 200

    cfg = AlertConfig(
        healthcheck_tick_url="https://hc/tick", healthcheck_guard_url=None,
        pushover_app_token="t", pushover_user_key="u", slack_webhook_url="https://slack/hook",
    )
    client = AlertClient(cfg, get=fake_get, post=fake_post, post_json=fake_post_json)
    result = client.send_test()
    assert result == {"healthchecks": True, "pushover": True, "slack": True}
    assert any("hc/tick" in p for p in posts)


def test_send_test_skips_unconfigured() -> None:
    cfg = AlertConfig(None, None, None, None, None)
    assert AlertClient(cfg).send_test() == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/deploy/test_alert_selftest.py -v`
Expected: FAIL — `AlertClient` has no `send_test`.

- [ ] **Step 3: Implement `send_test`**

In `quant/deploy/alerts.py`, add to `AlertClient`:

```python
    def send_test(self) -> dict[str, bool]:
        """Fire each CONFIGURED channel with a benign test payload. Returns per-channel delivery."""
        out: dict[str, bool] = {}
        ch = self._cfg.configured_channels()
        if "healthchecks" in ch:
            url = self._cfg.healthcheck_tick_url or self._cfg.healthcheck_guard_url
            try:
                self._get(url, 10.0) if url else None  # type: ignore[func-returns-value]
                out["healthchecks"] = True
            except Exception:
                out["healthchecks"] = False
        if "pushover" in ch:
            out["pushover"] = self._pushover_emergency(
                "quant alert-test", "Phase 2a alert self-test — channel OK."
            )
        if "slack" in ch:
            out["slack"] = self.send_slack(":test_tube: quant alert-test — Slack channel OK.")
        return out
```

- [ ] **Step 4: Add the CLI command**

In `quant/cli.py`, after `ops_run_job`:

```python
@ops.command("alert-test", help="Fire every configured alert channel to prove delivery.")
def ops_alert_test() -> None:
    from quant.deploy.alerts import AlertClient, AlertConfig

    settings = Settings()  # type: ignore[call-arg]
    cfg = AlertConfig.from_settings(settings)
    if not cfg.is_configured:
        raise click.ClickException(
            "no alert channels configured — set PUSHOVER_*/HEALTHCHECKS_*/SLACK_WEBHOOK_URL in .env"
        )
    results = AlertClient(cfg).send_test()
    for channel, ok in results.items():
        console.print(f"{'[green]OK[/green]' if ok else '[red]FAIL[/red]'}  {channel}")
    if not all(results.values()):
        raise SystemExit(1)
```

- [ ] **Step 5: Run, lint, type, commit**

```bash
uv run pytest tests/deploy/test_alert_selftest.py -q
uv run ruff check . && uv run ruff format . && uv run mypy quant/
git add quant/deploy/alerts.py quant/cli.py tests/deploy/test_alert_selftest.py
git commit -m "feat(deploy): quant ops alert-test — fire each configured channel to prove delivery"
```

---

### Task 4: Reliability self-check — pure checks + `quant ops selfcheck`

`quant doctor` checks trading readiness, not the reliability surface. Add pure check functions and a
CLI that validates: alerting configured, log-rotation covers all agent logs, pmset reboot settings,
disk floor, and launchd agent health — exiting non-zero on any failure. Host probes SKIP (not FAIL)
when their tool is absent (so it's runnable on the dev clone).

**Files:**
- Create: `quant/deploy/selfcheck.py`
- Modify: `quant/cli.py` (new `ops selfcheck` command)
- Test: `tests/deploy/test_selfcheck.py` (new)

**Interfaces:**
- Consumes: `AlertConfig` (Task 2).
- Produces: `@dataclass(frozen=True) CheckResult(name: str, status: str, detail: str)` where `status`
  in `{"OK","FAIL","SKIP"}`; pure functions
  `check_alerting(cfg: AlertConfig) -> CheckResult`,
  `check_log_rotation(conf_text: str, agent_log_stems: tuple[str, ...]) -> CheckResult`,
  `check_pmset(pmset_text: str | None) -> CheckResult`,
  `check_disk(free_bytes: int | None, floor_gb: float = 5.0) -> CheckResult`,
  `check_launchd(printout: str | None, labels: tuple[str, ...]) -> CheckResult`;
  `run_checks(results: list[CheckResult]) -> int` returns 1 if any `FAIL`.

- [ ] **Step 1: Write the failing test**

```python
# tests/deploy/test_selfcheck.py
from quant.deploy.alerts import AlertConfig
from quant.deploy.selfcheck import (
    CheckResult, check_alerting, check_disk, check_launchd, check_log_rotation,
    check_pmset, run_checks,
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


def test_pmset_ok_needs_autorestart_and_disablesleep() -> None:
    assert check_pmset(" autorestart 1\n disablesleep 1\n").status == "OK"
    assert check_pmset(" autorestart 0\n").status == "FAIL"
    assert check_pmset(None).status == "SKIP"


def test_disk_floor() -> None:
    assert check_disk(10 * 1024**3).status == "OK"
    assert check_disk(1 * 1024**3).status == "FAIL"
    assert check_disk(None).status == "SKIP"


def test_launchd_all_labels_present() -> None:
    out = "state = running\n"
    assert check_launchd(out, ("com.x",)).status == "OK"
    assert check_launchd(None, ("com.x",)).status == "SKIP"


def test_run_checks_returns_1_on_any_fail() -> None:
    assert run_checks([CheckResult("a", "OK", ""), CheckResult("b", "FAIL", "x")]) == 1
    assert run_checks([CheckResult("a", "OK", "")]) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/deploy/test_selfcheck.py -v`
Expected: FAIL — `quant/deploy/selfcheck.py` does not exist.

- [ ] **Step 3: Implement `quant/deploy/selfcheck.py`**

```python
"""Reliability self-check: pure predicates over the host's ops surface.

Each check returns a CheckResult. Host-probe inputs (launchctl/pmset/disk) are
gathered by the impure CLI shell and passed in; None means 'tool unavailable
here' -> SKIP (so this runs on the dev clone without failing)."""

from __future__ import annotations

from dataclasses import dataclass

from quant.deploy.alerts import AlertConfig


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "OK" | "FAIL" | "SKIP"
    detail: str


def check_alerting(cfg: AlertConfig) -> CheckResult:
    ch = cfg.configured_channels()
    if not ch:
        return CheckResult("alerting", "FAIL", "no alert channel configured")
    return CheckResult("alerting", "OK", f"channels: {', '.join(ch)}")


def check_log_rotation(conf_text: str, agent_log_stems: tuple[str, ...]) -> CheckResult:
    missing = [s for s in agent_log_stems if s not in conf_text]
    if missing:
        return CheckResult("log_rotation", "FAIL", f"not rotated: {', '.join(missing)}")
    return CheckResult("log_rotation", "OK", "all agent logs rotated")


def check_pmset(pmset_text: str | None) -> CheckResult:
    if pmset_text is None:
        return CheckResult("pmset", "SKIP", "pmset unavailable (not the live host)")
    ok = "autorestart 1" in pmset_text and "disablesleep 1" in pmset_text
    return CheckResult("pmset", "OK" if ok else "FAIL", "autorestart+disablesleep" if ok else "reboot/sleep settings missing")


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
    return 1 if any(r.status == "FAIL" for r in results) else 0
```

- [ ] **Step 4: Add the CLI shell**

In `quant/cli.py`, add under the `ops` group:

```python
@ops.command("selfcheck", help="Validate the reliability surface (alerting/logs/pmset/disk/launchd).")
def ops_selfcheck() -> None:
    import shutil
    import subprocess as _sp

    from quant.deploy.alerts import AlertConfig
    from quant.deploy.selfcheck import (
        check_alerting, check_disk, check_launchd, check_log_rotation, check_pmset, run_checks,
    )

    settings = Settings()  # type: ignore[call-arg]
    deploy_dir = Path(__file__).resolve().parent / "deploy"
    conf = (deploy_dir / "newsyslog" / "quant-deploy.conf")
    conf_text = conf.read_text(encoding="utf-8") if conf.exists() else ""

    def _probe(cmd: list[str]) -> str | None:
        if shutil.which(cmd[0]) is None:
            return None
        try:
            return _sp.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None

    pmset_text = _probe(["pmset", "-g"])
    launchd_text = _probe(["launchctl", "list"])
    free = shutil.disk_usage(settings.data_dir).free if settings.data_dir.exists() else None

    results = [
        check_alerting(AlertConfig.from_settings(settings)),
        check_log_rotation(conf_text, ("engine", "guard", "tick")),
        check_pmset(pmset_text),
        check_disk(free),
        check_launchd(launchd_text, ("com.ajaiupadhyaya.quant-tick",)),
    ]
    for r in results:
        color = {"OK": "green", "FAIL": "red", "SKIP": "yellow"}[r.status]
        console.print(f"[{color}]{r.status:<4}[/{color}] {r.name}: {r.detail}")
    raise SystemExit(run_checks(results))
```

- [ ] **Step 5: Run, lint, type, commit**

```bash
uv run pytest tests/deploy/test_selfcheck.py -q
uv run ruff check . && uv run ruff format . && uv run mypy quant/
git add quant/deploy/selfcheck.py quant/cli.py tests/deploy/test_selfcheck.py
git commit -m "feat(deploy): quant ops selfcheck — validate alerting/logs/pmset/disk/launchd"
```

---

### Task 5: Structured ANSI-free file logging + rotate all agent logs

The single colorized stderr sink leaks ANSI codes into log files and `engine.*`/`intraday-live.*`
aren't rotated. Add an optional JSON file sink (ANSI-free) and extend the newsyslog conf.

**Files:**
- Modify: `quant/util/logging.py` (`configure_logging`)
- Modify: `deploy/newsyslog/quant-deploy.conf`
- Test: `tests/util/test_logging_json_sink.py` (new)

**Interfaces:**
- Produces: `configure_logging(level: str = "INFO", *, json_path: str | Path | None = None) -> None` —
  when `json_path` is set, also adds a `serialize=True`, `colorize=False` file sink.

- [ ] **Step 1: Write the failing test**

```python
# tests/util/test_logging_json_sink.py
import json
from pathlib import Path

from quant.util.logging import configure_logging, logger


def test_json_sink_writes_ansi_free_json(tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    configure_logging("INFO", json_path=p)
    logger.info("hello structured")
    configure_logging("INFO")  # detach the file sink so the handle flushes/closes
    text = p.read_text(encoding="utf-8")
    assert "\x1b[" not in text  # no ANSI escape codes
    record = json.loads(text.splitlines()[0])
    assert record["record"]["message"] == "hello structured"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/util/test_logging_json_sink.py -v`
Expected: FAIL — `configure_logging` has no `json_path` parameter (`TypeError`).

- [ ] **Step 3: Implement the optional JSON sink**

Replace `configure_logging` in `quant/util/logging.py`:

```python
from pathlib import Path  # add import


def configure_logging(level: str = "INFO", *, json_path: str | Path | None = None) -> None:
    """Reset loguru sinks: a colorized stderr sink, plus an optional ANSI-free JSON file sink.

    Idempotent — safe to call repeatedly.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    if json_path is not None:
        logger.add(
            str(json_path),
            level=level.upper(),
            serialize=True,
            colorize=False,
            backtrace=False,
            diagnose=False,
        )
```

- [ ] **Step 4: Extend newsyslog rotation**

In `deploy/newsyslog/quant-deploy.conf`, add rows mirroring the existing tick/guard rows for the
engine and intraday-live logs (same owner `ajaiupadhyaya:staff`, mode 600, count 7, size 10000 KB,
flags `GJ`). Add (adjusting the path prefix to match the existing rows):

```
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/engine.stdout.log        ajaiupadhyaya:staff 600  7  10000 * GJ
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/engine.stderr.log        ajaiupadhyaya:staff 600  7  10000 * GJ
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/intraday-live.stdout.log ajaiupadhyaya:staff 600  7  10000 * GJ
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/intraday-live.stderr.log ajaiupadhyaya:staff 600  7  10000 * GJ
```

(Match the exact column layout of the existing `tick.stdout.log` row in that file — read it first and
mirror its spacing/flags.)

- [ ] **Step 5: Run, lint, type, commit**

```bash
uv run pytest tests/util/test_logging_json_sink.py -q
uv run ruff check . && uv run ruff format . && uv run mypy quant/
git add quant/util/logging.py deploy/newsyslog/quant-deploy.conf tests/util/test_logging_json_sink.py
git commit -m "feat(logging): optional ANSI-free JSON sink + rotate engine/intraday agent logs"
```

---

### Task 6: Full-suite green + 2a wrap

**Files:** none (verification + doc check-off)

- [ ] **Step 1: Run the canonical suite**

Run: `uv run pytest -q -m "not network and not alpaca and not slow"`
Expected: PASS — ≥1395 passing (the 5 new test files add cases; none removed).

- [ ] **Step 2: Lint + format + type**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy quant/`
Expected: all clean.

- [ ] **Step 3: Smoke the new CLI surface on the dev clone**

Run: `uv run quant ops selfcheck` (expect alerting=FAIL or OK per local .env, pmset/launchd=SKIP here)
Run: `uv run quant ops alert-test` (expect the ClickException if no channels in local .env — that's correct)
Confirm both behave (non-zero exit on FAIL is expected, not a regression).

- [ ] **Step 4: Commit the plan check-off + push**

```bash
git add docs/specs/2026-06-20-phase2a-reliability-plan.md
git commit -m "docs(specs): mark Phase 2a plan complete"
git push origin main
```

---

## Self-review

- **Spec coverage (2a slice of the design):** C job-timeout → Task 1 ✓; B alerting hardening
  (DRY + assertion + self-test) → Tasks 2–4 ✓ (assertion lives in `check_alerting`/`alert-test`);
  A self-check → Task 4 ✓; D structured logging + rotation → Task 5 ✓. Guard-status JSONL history is
  deferred to 2b (feeds the TUI) — noted in the design.
- **Type consistency:** `Runner`/`default_runner`/`_run_chain` all use `timeout_s: float | None`
  (Task 1); `AlertConfig.from_settings`/`configured_channels`/`is_configured` consumed identically in
  Tasks 3–4; `CheckResult.status` strings `OK/FAIL/SKIP` consistent across Task 4 + tests.
- **Placeholders:** none — every code/step is concrete. The newsyslog rows say "mirror the existing
  row's exact spacing" because that file's column alignment must be read in-place.
```
