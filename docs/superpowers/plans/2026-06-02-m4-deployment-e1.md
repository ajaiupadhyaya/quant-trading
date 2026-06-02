# M4 Deployment E1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the M4 Mac mini the sole always-on host for the quant-trading daily system — a launchd-supervised, DST-correct, catch-up-aware local tick scheduler replacing GitHub Actions cron, plus the durability/idempotency fixes and off-box alerting that make M4-only hosting safe.

**Architecture:** A **pure decision core** (`quant/deploy/`: `calendar_clock`, `manifest`, `scheduler`, `markers`) with an injected clock, behind a **thin impure shell** (`dispatcher` + `quant ops` CLI) that does all I/O. launchd runs the dispatcher every 60s (`StartInterval`) and supervises `quant guard run` (`KeepAlive`). Correctness fixes harden the existing trade path (deterministic `client_order_id`, pre-submit marker, atomic fail-closed `halt.json`). Alerting is Pushover (halts) + healthchecks.io (dead-man's-switch).

**Tech Stack:** Python 3.12, `uv`, Click, pydantic-settings, `tomllib`, stdlib `zoneinfo`/`datetime`/`fcntl`, `requests` (already a dep); pytest + ruff + mypy strict. macOS launchd/pmset/newsyslog. Reuses existing `quant/util/trading_calendar.py`, `quant/governance/halt.py`, `quant/monitor/`, `quant/live/rebalance.py`, `quant/execution/`.

**Spec:** `docs/superpowers/specs/2026-06-02-m4-deployment-e1-design.md`.

**Commands** (per "always use uv"): single test `uv run pytest <path>::<name> -v`; lint `uv run ruff check .`; format-check `uv run ruff format --check .`; types `uv run mypy quant`. Markers: `network`, `alpaca`, `slow` (`--strict-markers` on). mypy excludes `tests/`.

---

## Shared types (defined once; every task must match these signatures exactly)

```python
# quant/util/atomic.py
def write_json_atomic(path: Path, payload: object) -> None
def atomic_write_text(path: Path, text: str) -> None

# quant/deploy/calendar_clock.py
ET: ZoneInfo                                  # America/New_York
NORMAL_CLOSE: time                            # 16:00
EARLY_CLOSE: time                             # 13:00
def to_et(now_utc: datetime) -> datetime
def is_trading_day(d: date) -> bool           # re-export of trading_calendar.is_trading_day
def session_close_et(d: date) -> time         # NORMAL_CLOSE, or EARLY_CLOSE on early-close days

# quant/deploy/manifest.py
class DayRule(StrEnum): WEEKDAYS_TRADING; SATURDAY; TRADING_DAY_EVENING
class CatchUpPolicy(StrEnum): SAME_DAY; NONE
@dataclass(frozen=True)
class Job:
    name: str
    trigger_et: time | None            # fixed ET trigger (None if close-relative)
    close_offset_min: int | None       # trigger = session_close - this (None if fixed)
    days: DayRule
    catch_up: CatchUpPolicy
    max_lateness: time                 # catch-up horizon end time (ET)
    max_lateness_next_day: bool        # horizon end is on session_date + 1
    max_runtime_s: int                 # stale-lock timeout for this job
    timing_critical: bool
    commands: tuple[tuple[str, ...], ...]   # each inner tuple = args after "quant"
    commit_paths: tuple[str, ...]
@dataclass(frozen=True)
class Manifest:
    jobs: tuple[Job, ...]
def load_manifest(path: Path) -> Manifest

# quant/deploy/markers.py
def marker_path(data_dir: Path, job_name: str, session_date: date) -> Path
def read_markers(data_dir: Path) -> dict[str, date]      # job_name -> latest session_date
def write_marker(data_dir: Path, job_name: str, session_date: date, *,
                 kind: str, fired_at_utc: datetime, exit_code: int, duration_s: float) -> Path

# quant/deploy/scheduler.py
class DispatchKind(StrEnum): FRESH; CATCH_UP; MISSED; MISSED_CRITICAL
@dataclass(frozen=True)
class Dispatch:
    job: Job
    kind: DispatchKind
    session_date: date
def due_jobs(now_et: datetime, manifest: Manifest, markers: Mapping[str, date]) -> list[Dispatch]

# quant/deploy/alerts.py
@dataclass(frozen=True)
class AlertConfig:
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None
class AlertClient:
    def __init__(self, config: AlertConfig, post=..., get=...) -> None
    def ping_success(self, url: str | None) -> None
    def ping_fail(self, url: str | None, body: str = "") -> None
    def send_emergency(self, title: str, message: str) -> bool
```

`FRESH_TOL_MIN = 3` (general FRESH window width). Timing-critical FRESH window = `[close−5min, close−2min]`; hard cutoff = `close−2min`.

---

## File structure

**New (`quant/`):** `util/atomic.py`, `deploy/__init__.py`, `deploy/calendar_clock.py`, `deploy/manifest.py`, `deploy/jobs.toml`, `deploy/markers.py`, `deploy/scheduler.py`, `deploy/alerts.py`, `deploy/dispatcher.py`.
**New (`deploy/` repo root):** `com.ajaiupadhyaya.quant-tick.plist`, `com.ajaiupadhyaya.quant-guard.plist`, `install.sh`, `uninstall.sh`, `pmset.sh`, `newsyslog/quant-deploy.conf`, `README.md`.
**New tests:** `tests/util/test_atomic.py`, `tests/deploy/__init__.py`, `test_calendar_clock.py`, `test_manifest.py`, `test_markers.py`, `test_scheduler.py`, `test_alerts.py`, `test_dispatcher.py`.
**Modified:** `quant/util/config.py`, `quant/governance/halt.py`, `quant/monitor/status.py`, `quant/execution/orders.py`, `quant/execution/alpaca.py`, `quant/live/rebalance.py`, `quant/monitor/daemon.py`, `quant/cli.py`, `.github/workflows/{premarket-health,daily-rebalance,posttrade-reconciliation,nightly-backtest,weekly-grid-search,weekly-validation-governance}.yml`.

---

## Task 1: Atomic-write utility

**Files:**
- Create: `quant/util/atomic.py`
- Test: `tests/util/test_atomic.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for atomic JSON/text writes."""

from __future__ import annotations

import json
from pathlib import Path

from quant.util.atomic import atomic_write_text, write_json_atomic


def test_write_json_atomic_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "x.json"
    write_json_atomic(p, {"b": 2, "a": 1})
    assert json.loads(p.read_text()) == {"a": 1, "b": 2}  # sorted keys, parses


def test_write_json_atomic_leaves_no_tmp_file(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    write_json_atomic(p, {"k": 1})
    siblings = list(tmp_path.iterdir())
    assert siblings == [p], f"stray temp files: {siblings}"


def test_atomic_write_text_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "y.txt"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text() == "second"
    assert list(tmp_path.iterdir()) == [p]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/util/test_atomic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.util.atomic'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Atomic file writes: write to a sibling temp file, then os.replace.

os.replace is an atomic rename on POSIX, so a reader never sees a half-written
file and a crash mid-write leaves either the old file or nothing — never a
truncated file. This is the durability primitive the marker/halt/status writers
use. (Mirrors the proven tmp+replace pattern in quant/intraday/data/store.py.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/util/test_atomic.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + types + commit**

```bash
uv run ruff format quant/util/atomic.py tests/util/test_atomic.py
uv run ruff check quant/util/atomic.py && uv run mypy quant/util/atomic.py
git add quant/util/atomic.py tests/util/test_atomic.py
git commit -m "feat(util): atomic write_json_atomic / atomic_write_text helper"
```

---

## Task 2: Alert settings in config

**Files:**
- Modify: `quant/util/config.py` (after `fred_api_key`, line 41)
- Test: `tests/util/test_config_alerts.py`

- [ ] **Step 1: Write the failing test**

```python
"""Alert-channel settings are optional so CI's dummy env still constructs Settings."""

from __future__ import annotations

import pytest

from quant.util.config import Settings


def test_alert_settings_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HEALTHCHECKS_TICK_URL", "HEALTHCHECKS_GUARD_URL",
              "PUSHOVER_APP_TOKEN", "PUSHOVER_USER_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FRED_API_KEY", "f")
    s = Settings()  # type: ignore[call-arg]
    assert s.healthcheck_tick_url is None
    assert s.pushover_app_token is None


def test_alert_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FRED_API_KEY", "f")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "tok")
    monkeypatch.setenv("HEALTHCHECKS_TICK_URL", "https://hc-ping.com/abc")
    s = Settings()  # type: ignore[call-arg]
    assert s.pushover_app_token == "tok"
    assert s.healthcheck_tick_url == "https://hc-ping.com/abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/util/test_config_alerts.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'healthcheck_tick_url'`.

- [ ] **Step 3: Add the fields**

In `quant/util/config.py`, immediately after the `fred_api_key` field (line 41), add:

```python
    # Off-box alerting (E1). Optional so CI's dummy env still constructs Settings.
    healthcheck_tick_url: str | None = Field(default=None, description="healthchecks.io tick ping URL")
    healthcheck_guard_url: str | None = Field(default=None, description="healthchecks.io guard ping URL")
    pushover_app_token: str | None = Field(default=None, description="Pushover application token")
    pushover_user_key: str | None = Field(default=None, description="Pushover user key")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/util/test_config_alerts.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Update .env.example + commit**

Append to `.env.example`:
```
# Off-box alerting (E1 deploy)
HEALTHCHECKS_TICK_URL=
HEALTHCHECKS_GUARD_URL=
PUSHOVER_APP_TOKEN=
PUSHOVER_USER_KEY=
```

```bash
uv run ruff check quant/util/config.py && uv run mypy quant/util/config.py
git add quant/util/config.py tests/util/test_config_alerts.py .env.example
git commit -m "feat(config): optional Pushover + healthchecks alert settings"
```

---

## Task 3: Atomic + fail-closed halt; atomic status

**Files:**
- Modify: `quant/governance/halt.py` (`load_halt` line 22-35; `_write` line 58-73)
- Modify: `quant/monitor/status.py` (`write_status` line 42)
- Test: `tests/governance/test_halt_durability.py`

- [ ] **Step 1: Write the failing test**

```python
"""Halt artifact must be written atomically and FAIL CLOSED on corruption."""

from __future__ import annotations

from pathlib import Path

from quant.governance.halt import halt_path, load_halt, set_halt


def test_corrupt_halt_reads_as_active(tmp_path: Path) -> None:
    # A truncated/garbage halt.json must be treated as HALTED, not raise / not open.
    p = halt_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json", encoding="utf-8")
    state = load_halt(tmp_path)
    assert state.active is True
    assert "corrupt" in state.reason.lower()


def test_non_dict_halt_reads_as_active(tmp_path: Path) -> None:
    p = halt_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_halt(tmp_path).active is True


def test_set_halt_leaves_no_tmp_file(tmp_path: Path) -> None:
    set_halt(tmp_path, reason="test")
    files = sorted(p.name for p in (tmp_path / "governance").iterdir())
    assert files == ["halt.json"], f"stray temp files: {files}"


def test_valid_halt_roundtrips(tmp_path: Path) -> None:
    set_halt(tmp_path, reason="drift breach")
    s = load_halt(tmp_path)
    assert s.active is True and s.reason == "drift breach"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/governance/test_halt_durability.py -v`
Expected: FAIL — `test_corrupt_halt_reads_as_active` raises `json.JSONDecodeError` (current `load_halt` does not catch it).

- [ ] **Step 3: Make `load_halt` fail-closed and `_write` atomic**

In `quant/governance/halt.py`, replace `load_halt` (lines 22-35) with:

```python
def load_halt(data_dir: Path) -> HaltState:
    path = halt_path(data_dir)
    if not path.exists():
        return HaltState(
            active=False, reason="not halted", updated_at=datetime.fromtimestamp(0, UTC)
        )
    # Fail closed: any unreadable/malformed halt artifact is treated as ACTIVE.
    # A corrupt halt.json must never read as "not halted" — that would let
    # trading proceed through a halt the artifact was meant to enforce.
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("halt artifact is not a JSON object")
        return HaltState(
            active=bool(raw.get("active", False)),
            reason=str(raw.get("reason", "")),
            updated_at=datetime.fromisoformat(str(raw["updated_at"])),
        )
    except (ValueError, KeyError, OSError) as exc:
        return HaltState(active=True, reason=f"corrupt halt artifact: {exc}", updated_at=datetime.now(UTC))
```

Replace `_write` (lines 58-73) with an atomic write:

```python
def _write(data_dir: Path, state: HaltState) -> None:
    from quant.util.atomic import write_json_atomic

    write_json_atomic(
        halt_path(data_dir),
        {
            "active": state.active,
            "reason": state.reason,
            "updated_at": state.updated_at.isoformat(),
        },
    )
```

- [ ] **Step 4: Make `write_status` atomic**

In `quant/monitor/status.py`, replace line 42 (`path.write_text(...)`) with:

```python
    from quant.util.atomic import write_json_atomic

    write_json_atomic(path, payload)
```

(Keep the `path = monitor_status_path(data_dir)` / `payload = {...}` lines; the `mkdir` in line 30 is now redundant but harmless — leave it or delete it. `write_status` still `return path`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/governance/test_halt_durability.py tests/monitor -v`
Expected: PASS (existing monitor/halt tests still green; new durability tests pass).

- [ ] **Step 6: Lint + types + commit**

```bash
uv run ruff format quant/governance/halt.py quant/monitor/status.py tests/governance/test_halt_durability.py
uv run ruff check quant/governance/halt.py quant/monitor/status.py && uv run mypy quant/governance/halt.py quant/monitor/status.py
git add quant/governance/halt.py quant/monitor/status.py tests/governance/test_halt_durability.py
git commit -m "fix(governance): atomic + fail-closed halt.json; atomic monitor_status"
```

---

## Task 4: `calendar_clock` (pure)

**Files:**
- Create: `quant/deploy/__init__.py` (empty), `quant/deploy/calendar_clock.py`
- Test: `tests/deploy/__init__.py` (empty), `tests/deploy/test_calendar_clock.py`

- [ ] **Step 1: Write the failing test**

```python
"""Pure ET clock + session-close helpers (DST + early-close)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

from quant.deploy.calendar_clock import (
    EARLY_CLOSE,
    NORMAL_CLOSE,
    is_trading_day,
    session_close_et,
    to_et,
)


def test_to_et_winter_is_utc_minus_5() -> None:
    # 2026-01-15 20:00 UTC -> 15:00 EST
    et = to_et(datetime(2026, 1, 15, 20, 0, tzinfo=UTC))
    assert (et.hour, et.minute) == (15, 0)


def test_to_et_summer_is_utc_minus_4() -> None:
    # 2026-07-15 19:55 UTC -> 15:55 EDT
    et = to_et(datetime(2026, 7, 15, 19, 55, tzinfo=UTC))
    assert (et.hour, et.minute) == (15, 55)


def test_to_et_accepts_naive_as_utc() -> None:
    et = to_et(datetime(2026, 7, 15, 19, 55))
    assert (et.hour, et.minute) == (15, 55)


def test_session_close_normal_day() -> None:
    assert session_close_et(date(2026, 6, 2)) == NORMAL_CLOSE == time(16, 0)


def test_session_close_early_close_day_after_thanksgiving() -> None:
    # 2026 Thanksgiving = 4th Thu Nov = Nov 26; day after = Nov 27 (early close).
    assert session_close_et(date(2026, 11, 27)) == EARLY_CLOSE == time(13, 0)


def test_is_trading_day_reexport() -> None:
    assert is_trading_day(date(2026, 6, 2)) is True
    assert is_trading_day(date(2026, 6, 6)) is False  # Saturday
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_calendar_clock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.deploy.calendar_clock'`.

- [ ] **Step 3: Implement**

`quant/deploy/__init__.py`: empty file. `tests/deploy/__init__.py`: empty file.

`quant/deploy/calendar_clock.py`:

```python
"""Pure ET wall-clock + session-close classification for the tick scheduler.

UTC has no DST, so converting an aware UTC instant to America/New_York is always
unambiguous — this is what makes the scheduler DST-correct (the old GitHub crons
were fixed-UTC and drifted +1h in winter). Trading-day / early-close facts are
delegated to quant/util/trading_calendar.py (single source of truth). No I/O and
no datetime.now() here — the current time is always an argument.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from quant.util.trading_calendar import is_early_close, is_trading_day

ET = ZoneInfo("America/New_York")
NORMAL_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

__all__ = ["ET", "NORMAL_CLOSE", "EARLY_CLOSE", "to_et", "is_trading_day", "session_close_et"]


def to_et(now_utc: datetime) -> datetime:
    """Convert an instant to America/New_York. A naive datetime is assumed UTC."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    return now_utc.astimezone(ET)


def session_close_et(d: date) -> time:
    """NYSE close time (ET) for trading day ``d``: 13:00 on early-close days, else 16:00.

    Raises ValueError on a non-trading day (callers gate on the day rule first).
    """
    if not is_trading_day(d):
        raise ValueError(f"{d} is not a trading day")
    return EARLY_CLOSE if is_early_close(d) else NORMAL_CLOSE
```

(`is_trading_day` is re-exported via the import + `__all__`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/deploy/test_calendar_clock.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint + types + commit**

```bash
uv run ruff format quant/deploy/ tests/deploy/
uv run ruff check quant/deploy/calendar_clock.py && uv run mypy quant/deploy/calendar_clock.py
git add quant/deploy/__init__.py quant/deploy/calendar_clock.py tests/deploy/__init__.py tests/deploy/test_calendar_clock.py
git commit -m "feat(deploy): pure ET calendar_clock with session_close_et"
```

---

## Task 5: Job manifest + `jobs.toml`

**Files:**
- Create: `quant/deploy/manifest.py`, `quant/deploy/jobs.toml`
- Test: `tests/deploy/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
"""Manifest loader: parse + validate jobs.toml."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest

from quant.deploy.manifest import CatchUpPolicy, DayRule, load_manifest

REPO_MANIFEST = Path(__file__).resolve().parents[2] / "quant" / "deploy" / "jobs.toml"


def test_loads_six_jobs() -> None:
    m = load_manifest(REPO_MANIFEST)
    assert {j.name for j in m.jobs} == {
        "premarket-health", "daily-rebalance", "posttrade-reconciliation",
        "nightly-backtest", "weekly-grid-search", "weekly-validation-governance",
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
    assert ("rebalance",) in reb.commands


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.deploy.manifest'`.

- [ ] **Step 3: Implement `manifest.py`**

```python
"""Version-controlled job manifest: the single source of truth that replaces the
six GitHub Actions cron schedules. Editing jobs.toml is how the schedule changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import time
from enum import StrEnum
from pathlib import Path


class DayRule(StrEnum):
    WEEKDAYS_TRADING = "WEEKDAYS_TRADING"
    SATURDAY = "SATURDAY"
    TRADING_DAY_EVENING = "TRADING_DAY_EVENING"


class CatchUpPolicy(StrEnum):
    SAME_DAY = "SAME_DAY"
    NONE = "NONE"


@dataclass(frozen=True)
class Job:
    name: str
    trigger_et: time | None
    close_offset_min: int | None
    days: DayRule
    catch_up: CatchUpPolicy
    max_lateness: time
    max_lateness_next_day: bool
    max_runtime_s: int
    timing_critical: bool
    commands: tuple[tuple[str, ...], ...]
    commit_paths: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    jobs: tuple[Job, ...]


def _parse_time(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def load_manifest(path: Path) -> Manifest:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    jobs: list[Job] = []
    for entry in raw.get("job", []):
        trigger_et = _parse_time(entry["trigger_et"]) if "trigger_et" in entry else None
        close_offset = entry.get("close_offset_min")
        if (trigger_et is None) == (close_offset is None):
            raise ValueError(f"job {entry.get('name')!r}: set exactly one of trigger_et / close_offset_min")
        commands = tuple(tuple(str(a) for a in c) for c in entry["commands"])
        if not commands or not all(isinstance(c, tuple) and c for c in commands):
            raise ValueError(f"job {entry.get('name')!r}: commands must be a non-empty list of arg-lists")
        jobs.append(
            Job(
                name=str(entry["name"]),
                trigger_et=trigger_et,
                close_offset_min=int(close_offset) if close_offset is not None else None,
                days=DayRule(entry["days"]),
                catch_up=CatchUpPolicy(entry["catch_up"]),
                max_lateness=_parse_time(entry["max_lateness"]),
                max_lateness_next_day=bool(entry.get("max_lateness_next_day", False)),
                max_runtime_s=int(entry["max_runtime_s"]),
                timing_critical=bool(entry.get("timing_critical", False)),
                commands=commands,
                commit_paths=tuple(str(p) for p in entry.get("commit_paths", [])),
            )
        )
    names = [j.name for j in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate job names in manifest: {names}")
    return Manifest(jobs=tuple(jobs))
```

- [ ] **Step 4: Create `quant/deploy/jobs.toml`**

```toml
# Source of truth for the M4 tick scheduler (replaces .github/workflows/*.yml crons).
# Times are America/New_York. commands = args appended after `uv run quant`.

[[job]]
name = "premarket-health"
trigger_et = "09:00"
days = "WEEKDAYS_TRADING"
catch_up = "SAME_DAY"
max_lateness = "14:00"
max_runtime_s = 1200
commands = [
  ["data", "refresh", "--start", "2018-01-01"],
  ["doctor"],
  ["governance", "status"],
  ["rebalance", "--dry-run"],
]
commit_paths = []

[[job]]
name = "daily-rebalance"
close_offset_min = 5
days = "WEEKDAYS_TRADING"
catch_up = "NONE"
timing_critical = true
max_lateness = "16:00"
max_runtime_s = 600
commands = [
  ["data", "refresh", "--start", "2018-01-01"],
  ["risk", "pretrade"],
  ["doctor"],
  ["rebalance"],
]
commit_paths = ["data/live/", "data/raw/", "data/ops/health/", "data/risk/"]

[[job]]
name = "posttrade-reconciliation"
trigger_et = "17:30"
days = "WEEKDAYS_TRADING"
catch_up = "SAME_DAY"
max_lateness = "23:59"
max_runtime_s = 600
commands = [["__script__", "scripts/reconcile_live.py"]]
commit_paths = ["docs/live-recon/"]

[[job]]
name = "nightly-backtest"
trigger_et = "22:00"
days = "TRADING_DAY_EVENING"
catch_up = "SAME_DAY"
max_lateness = "09:00"
max_lateness_next_day = true
max_runtime_s = 7200
commands = [["__backtest_matrix__", "--quick"]]
commit_paths = ["data/backtests/"]

[[job]]
name = "weekly-grid-search"
trigger_et = "02:00"
days = "SATURDAY"
catch_up = "SAME_DAY"
max_lateness = "23:59"
max_lateness_next_day = true
max_runtime_s = 28800
commands = [["__backtest_matrix__", "--full"]]
commit_paths = ["data/backtests/"]

[[job]]
name = "weekly-validation-governance"
trigger_et = "04:00"
days = "SATURDAY"
catch_up = "SAME_DAY"
max_lateness = "23:59"
max_lateness_next_day = true
max_runtime_s = 28800
commands = [["__validate_matrix__"], ["governance", "refresh"], ["governance", "status"]]
commit_paths = ["data/backtests/", "data/governance/"]
```

> Note: `__script__`, `__backtest_matrix__`, `__validate_matrix__` are sentinels the dispatcher expands (Task 12) — the matrix jobs loop over strategy slugs (mirroring the GH `for slug in ...` loops). The exact slug list + per-job flags are encoded in the dispatcher's expansion so the manifest stays declarative.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/deploy/test_manifest.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Lint + types + commit**

```bash
uv run ruff format quant/deploy/manifest.py tests/deploy/test_manifest.py
uv run ruff check quant/deploy/manifest.py && uv run mypy quant/deploy/manifest.py
git add quant/deploy/manifest.py quant/deploy/jobs.toml tests/deploy/test_manifest.py
git commit -m "feat(deploy): job manifest model + jobs.toml (6 jobs, ET, DST-correct)"
```

---

## Task 6: Idempotency markers

**Files:**
- Create: `quant/deploy/markers.py`
- Test: `tests/deploy/test_markers.py`

- [ ] **Step 1: Write the failing test**

```python
"""Session-scoped idempotency markers under data/ops/scheduler/."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from quant.deploy.markers import marker_path, read_markers, write_marker


def test_write_then_read_latest(tmp_path: Path) -> None:
    write_marker(tmp_path, "daily-rebalance", date(2026, 6, 1),
                 kind="FRESH", fired_at_utc=datetime(2026, 6, 1, 19, 55, tzinfo=UTC),
                 exit_code=0, duration_s=4.2)
    write_marker(tmp_path, "daily-rebalance", date(2026, 6, 2),
                 kind="FRESH", fired_at_utc=datetime(2026, 6, 2, 19, 55, tzinfo=UTC),
                 exit_code=0, duration_s=4.0)
    assert read_markers(tmp_path)["daily-rebalance"] == date(2026, 6, 2)


def test_read_markers_empty_when_no_dir(tmp_path: Path) -> None:
    assert read_markers(tmp_path) == {}


def test_marker_write_is_atomic_no_tmp(tmp_path: Path) -> None:
    write_marker(tmp_path, "premarket-health", date(2026, 6, 2),
                 kind="FRESH", fired_at_utc=datetime(2026, 6, 2, 13, 0, tzinfo=UTC),
                 exit_code=0, duration_s=1.0)
    d = tmp_path / "ops" / "scheduler"
    assert all(p.suffix == ".json" for p in d.iterdir()), list(d.iterdir())


def test_marker_path_encodes_job_and_date(tmp_path: Path) -> None:
    p = marker_path(tmp_path, "daily-rebalance", date(2026, 6, 2))
    assert p.name == "daily-rebalance.2026-06-02.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_markers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.deploy.markers'`.

- [ ] **Step 3: Implement**

```python
"""Once-per-session idempotency markers + per-day run-ledger.

A marker file per (job, session-date) is written ONLY after a job's command
chain completes (the timing-critical pre-submit marker is the one exception,
written by quant rebalance itself before order submission). The scheduler keys
on session-date — not "today" — so a missed prior session is representable.
Markers are host-local run state (git-ignored), not committed artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from quant.util.atomic import write_json_atomic


def _scheduler_dir(data_dir: Path) -> Path:
    return data_dir / "ops" / "scheduler"


def marker_path(data_dir: Path, job_name: str, session_date: date) -> Path:
    return _scheduler_dir(data_dir) / f"{job_name}.{session_date.isoformat()}.json"


def write_marker(
    data_dir: Path,
    job_name: str,
    session_date: date,
    *,
    kind: str,
    fired_at_utc: datetime,
    exit_code: int,
    duration_s: float,
) -> Path:
    path = marker_path(data_dir, job_name, session_date)
    write_json_atomic(
        path,
        {
            "job": job_name,
            "session_date": session_date.isoformat(),
            "kind": kind,
            "fired_at_utc": fired_at_utc.isoformat(),
            "exit_code": exit_code,
            "duration_s": duration_s,
        },
    )
    return path


def read_markers(data_dir: Path) -> dict[str, date]:
    """Return {job_name: latest session-date with a marker}."""
    d = _scheduler_dir(data_dir)
    if not d.exists():
        return {}
    latest: dict[str, date] = {}
    for f in d.glob("*.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            job = str(raw["job"])
            sd = date.fromisoformat(str(raw["session_date"]))
        except (ValueError, KeyError, OSError):
            continue
        if job not in latest or sd > latest[job]:
            latest[job] = sd
    return latest


def is_marked(markers: Mapping[str, date], job_name: str, session_date: date) -> bool:
    return markers.get(job_name) == session_date
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/deploy/test_markers.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Gitignore markers + commit**

Append to `.gitignore`:
```
# E1 scheduler run-state (host-local, not committed)
data/ops/scheduler/
```

```bash
uv run ruff format quant/deploy/markers.py tests/deploy/test_markers.py
uv run ruff check quant/deploy/markers.py && uv run mypy quant/deploy/markers.py
git add quant/deploy/markers.py tests/deploy/test_markers.py .gitignore
git commit -m "feat(deploy): session-scoped idempotency markers (atomic, git-ignored)"
```

---

## Task 7: Scheduler core (`due_jobs`)

**Files:**
- Create: `quant/deploy/scheduler.py`
- Test: `tests/deploy/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
"""Pure due/catch-up/missed engine — the heart of the scheduler."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from quant.deploy.calendar_clock import ET
from quant.deploy.manifest import CatchUpPolicy, DayRule, Job, Manifest
from quant.deploy.scheduler import DispatchKind, due_jobs


def _et(y, mo, d, h, mi) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def _job(**kw) -> Job:
    base = dict(
        name="j", trigger_et=time(9, 0), close_offset_min=None,
        days=DayRule.WEEKDAYS_TRADING, catch_up=CatchUpPolicy.SAME_DAY,
        max_lateness=time(14, 0), max_lateness_next_day=False, max_runtime_s=600,
        timing_critical=False, commands=(("doctor",),), commit_paths=(),
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
    j = _job(name="reb", trigger_et=None, close_offset_min=5,
             catch_up=CatchUpPolicy.NONE, timing_critical=True, max_lateness=time(16, 0))
    out = due_jobs(_et(2026, 6, 2, 15, 56), _m(j), {})  # close 16:00 -> window 15:55-15:58
    assert [d.kind for d in out] == [DispatchKind.FRESH]


def test_timing_critical_missed_after_hard_cutoff() -> None:
    j = _job(name="reb", trigger_et=None, close_offset_min=5,
             catch_up=CatchUpPolicy.NONE, timing_critical=True, max_lateness=time(16, 0))
    out = due_jobs(_et(2026, 6, 2, 16, 5), _m(j), {})  # past close-2min
    assert [d.kind for d in out] == [DispatchKind.MISSED_CRITICAL]


def test_timing_critical_early_close_day() -> None:
    j = _job(name="reb", trigger_et=None, close_offset_min=5,
             catch_up=CatchUpPolicy.NONE, timing_critical=True, max_lateness=time(16, 0))
    # 2026-11-27 early close 13:00 -> rebalance window 12:55-12:58.
    assert [d.kind for d in due_jobs(_et(2026, 11, 27, 12, 56), _m(j), {})] == [DispatchKind.FRESH]
    # a 15:55 tick that day is past the 13:00 close -> MISSED_CRITICAL
    assert [d.kind for d in due_jobs(_et(2026, 11, 27, 15, 55), _m(j), {})] == [DispatchKind.MISSED_CRITICAL]


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
    j = _job(name="nb", days=DayRule.TRADING_DAY_EVENING, trigger_et=time(22, 0),
             max_lateness=time(9, 0), max_lateness_next_day=True)
    # Sat 01:30 ET, no marker -> Friday's (2026-06-05) backtest is caught up
    out = due_jobs(_et(2026, 6, 6, 1, 30), _m(j), {})
    assert len(out) == 1 and out[0].kind == DispatchKind.CATCH_UP
    assert out[0].session_date == date(2026, 6, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.deploy.scheduler'`.

- [ ] **Step 3: Implement**

```python
"""Pure scheduler: decide which manifest jobs are due this tick and as what kind.

Two disjoint ladders so a job is never in two classes:
  * catch-up-safe (catch_up=SAME_DAY):  FRESH -> CATCH_UP -> MISSED
  * timing-critical (catch_up=NONE):    FRESH -> MISSED_CRITICAL  (no catch-up)

Session attribution (`_session_date`) maps the tick to the trading session a job
serves; for TRADING_DAY_EVENING the session can be the prior calendar day when
ticking in the 00:00-09:00 catch-up window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from quant.deploy.calendar_clock import ET, session_close_et
from quant.deploy.manifest import DayRule, Job, Manifest
from quant.util.trading_calendar import is_trading_day, previous_trading_day

FRESH_TOL_MIN = 3
CRITICAL_CUTOFF_BEFORE_CLOSE_MIN = 2


class DispatchKind(StrEnum):
    FRESH = "FRESH"
    CATCH_UP = "CATCH_UP"
    MISSED = "MISSED"
    MISSED_CRITICAL = "MISSED_CRITICAL"


@dataclass(frozen=True)
class Dispatch:
    job: Job
    kind: DispatchKind
    session_date: date


def _session_date(job: Job, now_et: datetime) -> date | None:
    d = now_et.date()
    t = now_et.time()
    if job.days == DayRule.WEEKDAYS_TRADING:
        return d if is_trading_day(d) else None
    if job.days == DayRule.SATURDAY:
        return d if d.weekday() == 5 else None
    if job.days == DayRule.TRADING_DAY_EVENING:
        if t >= time(22, 0):
            return d if is_trading_day(d) else None
        if t <= time(9, 0):
            return previous_trading_day(d)  # always a trading day
        return None
    return None


def _trigger_time(job: Job, d: date) -> time:
    if job.close_offset_min is not None:
        close = datetime.combine(d, session_close_et(d))
        return (close - timedelta(minutes=job.close_offset_min)).time()
    assert job.trigger_et is not None
    return job.trigger_et


def _combine(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET)


def _classify(job: Job, now_et: datetime, d: date) -> DispatchKind | None:
    trigger = _trigger_time(job, d)
    # For the evening job the trigger datetime is on the SESSION date d (22:00),
    # while now_et may be d+1 in the morning. Combine on d.
    trig_dt = _combine(d, trigger)

    if job.timing_critical:
        close_dt = _combine(d, session_close_et(d))
        hard_cutoff = close_dt - timedelta(minutes=CRITICAL_CUTOFF_BEFORE_CLOSE_MIN)
        if now_et < trig_dt:
            return None
        if trig_dt <= now_et <= hard_cutoff:
            return DispatchKind.FRESH
        return DispatchKind.MISSED_CRITICAL

    fresh_end = trig_dt + timedelta(minutes=FRESH_TOL_MIN)
    horizon_day = d + timedelta(days=1) if job.max_lateness_next_day else d
    horizon_end = _combine(horizon_day, job.max_lateness)
    if now_et < trig_dt:
        return None
    if trig_dt <= now_et <= fresh_end:
        return DispatchKind.FRESH
    if fresh_end < now_et <= horizon_end:
        return DispatchKind.CATCH_UP
    return DispatchKind.MISSED


def due_jobs(
    now_et: datetime, manifest: Manifest, markers: Mapping[str, date]
) -> list[Dispatch]:
    out: list[Dispatch] = []
    for job in manifest.jobs:
        d = _session_date(job, now_et)
        if d is None:
            continue
        if markers.get(job.name) == d:
            continue  # already handled this session (any kind)
        kind = _classify(job, now_et, d)
        if kind is None:
            continue
        out.append(Dispatch(job=job, kind=kind, session_date=d))
    return out
```

> The ladder is selected by `job.timing_critical` (not by reading `CatchUpPolicy` in `scheduler.py`); the two always agree in `jobs.toml` (`NONE`↔`timing_critical=true`), and a manifest-validation test (Task 14) asserts that pairing — so `scheduler.py` does not import `CatchUpPolicy`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/deploy/test_scheduler.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Lint + types + commit**

```bash
uv run ruff format quant/deploy/scheduler.py tests/deploy/test_scheduler.py
uv run ruff check quant/deploy/scheduler.py && uv run mypy quant/deploy/scheduler.py
git add quant/deploy/scheduler.py tests/deploy/test_scheduler.py
git commit -m "feat(deploy): pure due_jobs scheduler (FRESH/CATCH_UP/MISSED/MISSED_CRITICAL)"
```

---

## Task 8: Alert client

**Files:**
- Create: `quant/deploy/alerts.py`
- Test: `tests/deploy/test_alerts.py`

- [ ] **Step 1: Write the failing test**

```python
"""AlertClient with an injected HTTP transport (no network in tests)."""

from __future__ import annotations

from quant.deploy.alerts import AlertClient, AlertConfig


class _Recorder:
    def __init__(self) -> None:
        self.gets: list[tuple[str, float]] = []
        self.posts: list[tuple[str, dict]] = []
        self.fail = False

    def get(self, url: str, timeout: float) -> int:
        if self.fail:
            raise OSError("network down")
        self.gets.append((url, timeout))
        return 200

    def post(self, url: str, data: dict, timeout: float) -> int:
        if self.fail:
            raise OSError("network down")
        self.posts.append((url, data))
        return 200


def _cfg() -> AlertConfig:
    return AlertConfig(
        healthcheck_tick_url="https://hc-ping.com/tick",
        healthcheck_guard_url=None,
        pushover_app_token="apptok",
        pushover_user_key="userkey",
    )


def test_ping_success_hits_url() -> None:
    r = _Recorder()
    AlertClient(_cfg(), get=r.get, post=r.post).ping_success("https://hc-ping.com/tick")
    assert r.gets and r.gets[0][0] == "https://hc-ping.com/tick"


def test_ping_success_none_url_is_noop() -> None:
    r = _Recorder()
    AlertClient(_cfg(), get=r.get, post=r.post).ping_success(None)
    assert r.gets == []


def test_ping_fail_appends_fail_path() -> None:
    r = _Recorder()
    AlertClient(_cfg(), get=r.get, post=r.post).ping_fail("https://hc-ping.com/tick", "boom")
    assert r.gets[0][0].endswith("/fail")


def test_send_emergency_builds_priority2_payload() -> None:
    r = _Recorder()
    ok = AlertClient(_cfg(), get=r.get, post=r.post).send_emergency("HALT", "drift breach")
    assert ok is True
    url, data = r.posts[0]
    assert "pushover" in url
    assert data["priority"] == 2 and data["retry"] == 60 and data["expire"] == 3600
    assert data["title"] == "HALT" and data["message"] == "drift breach"
    assert data["token"] == "apptok" and data["user"] == "userkey"


def test_send_emergency_returns_false_when_unconfigured() -> None:
    cfg = AlertConfig(None, None, None, None)
    assert AlertClient(cfg, get=_Recorder().get, post=_Recorder().post).send_emergency("x", "y") is False


def test_send_emergency_returns_false_on_network_error() -> None:
    r = _Recorder()
    r.fail = True
    assert AlertClient(_cfg(), get=r.get, post=r.post).send_emergency("x", "y") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_alerts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.deploy.alerts'`.

- [ ] **Step 3: Implement**

```python
"""Alerting: healthchecks.io liveness pings + Pushover emergency push.

HTTP is injected (get/post callables) so tests assert on calls without network.
Secret-bearing URLs (healthchecks ping URLs) are never logged — only outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import requests

from quant.util.logging import logger

GetFn = Callable[[str, float], int]
PostFn = Callable[[str, dict[str, object], float], int]

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def _default_get(url: str, timeout: float) -> int:
    return requests.get(url, timeout=timeout).status_code


def _default_post(url: str, data: dict[str, object], timeout: float) -> int:
    return requests.post(url, data=data, timeout=timeout).status_code


@dataclass(frozen=True)
class AlertConfig:
    healthcheck_tick_url: str | None
    healthcheck_guard_url: str | None
    pushover_app_token: str | None
    pushover_user_key: str | None


class AlertClient:
    def __init__(self, config: AlertConfig, *, get: GetFn = _default_get, post: PostFn = _default_post) -> None:
        self._cfg = config
        self._get = get
        self._post = post

    def ping_success(self, url: str | None) -> None:
        if not url:
            return
        try:
            self._get(url, 10.0)
        except Exception:  # liveness ping is best-effort; a gap is itself the signal
            logger.warning("healthcheck success ping failed (name suppressed)")

    def ping_fail(self, url: str | None, body: str = "") -> None:
        if not url:
            return
        try:
            self._get(url.rstrip("/") + "/fail", 10.0)
        except Exception:
            logger.warning("healthcheck fail ping failed (name suppressed)")

    def send_emergency(self, title: str, message: str) -> bool:
        """Pushover Emergency (priority 2) push. Returns True iff delivered."""
        if not (self._cfg.pushover_app_token and self._cfg.pushover_user_key):
            logger.error("emergency push requested but Pushover not configured: {}", title)
            return False
        payload: dict[str, object] = {
            "token": self._cfg.pushover_app_token,
            "user": self._cfg.pushover_user_key,
            "title": title,
            "message": message,
            "priority": 2,
            "retry": 60,
            "expire": 3600,
        }
        try:
            status = self._post(_PUSHOVER_URL, payload, 10.0)
        except Exception as exc:
            logger.error("emergency push failed to send: {!r}", exc)
            return False
        if status >= 400:
            logger.error("emergency push rejected: HTTP {}", status)
            return False
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/deploy/test_alerts.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint + types + commit**

```bash
uv run ruff format quant/deploy/alerts.py tests/deploy/test_alerts.py
uv run ruff check quant/deploy/alerts.py && uv run mypy quant/deploy/alerts.py
git add quant/deploy/alerts.py tests/deploy/test_alerts.py
git commit -m "feat(deploy): AlertClient (healthchecks pings + Pushover emergency)"
```

---

## Task 9: Deterministic `client_order_id`

**Files:**
- Modify: `quant/execution/orders.py` (`make_client_order_id`, line 29-35)
- Modify: `quant/execution/alpaca.py` (the `submit_order` call site that builds the COID — locate via `make_client_order_id(`)
- Test: `tests/execution/test_orders_deterministic.py`

- [ ] **Step 1: Write the failing test**

```python
"""client_order_id must be deterministic per (strategy, symbol, session-date)
so the broker rejects a duplicate same-day resubmission (idempotency)."""

from __future__ import annotations

from datetime import date

from quant.execution.orders import make_client_order_id


def test_client_order_id_is_deterministic() -> None:
    a = make_client_order_id("trend", "SPY", date(2026, 6, 2))
    b = make_client_order_id("trend", "SPY", date(2026, 6, 2))
    assert a == b


def test_client_order_id_differs_by_symbol_and_date() -> None:
    assert make_client_order_id("trend", "SPY", date(2026, 6, 2)) != \
           make_client_order_id("trend", "EFA", date(2026, 6, 2))
    assert make_client_order_id("trend", "SPY", date(2026, 6, 2)) != \
           make_client_order_id("trend", "SPY", date(2026, 6, 3))


def test_client_order_id_carries_slug_prefix_for_attribution() -> None:
    coid = make_client_order_id("multi-factor", "JPM", date(2026, 6, 2))
    assert coid.startswith("multi-factor-20260602-JPM")
    assert len(coid) <= 48  # Alpaca client_order_id length limit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_orders_deterministic.py -v`
Expected: FAIL — `test_client_order_id_is_deterministic` (current impl appends `uuid4().hex[:8]`, so two calls differ).

- [ ] **Step 3: Make it deterministic**

In `quant/execution/orders.py`, replace `make_client_order_id` (lines 29-35) with:

```python
def make_client_order_id(strategy_slug: str, symbol: str, dt: date) -> str:
    """Deterministic id: ``<slug>-<YYYYMMDD>-<symbol>``.

    Deterministic per (strategy, symbol, session-date) so a resubmission of the
    same logical order on the same day collides on client_order_id and Alpaca
    rejects the duplicate — broker-level idempotency that guards against a
    crash-then-retry double-submit. The slug prefix still attributes fills to a
    strategy. (Alpaca caps client_order_id at 48 chars; slugs+symbols here are
    well under that.)
    """
    return f"{strategy_slug}-{dt:%Y%m%d}-{symbol}"
```

Remove the now-unused `import uuid` (line 6) if nothing else in the file uses it.

- [ ] **Step 4: Verify the alpaca call site passes the session date**

Open `quant/execution/alpaca.py`, find the `make_client_order_id(` call inside `submit_order`. Confirm it passes `(order.strategy_slug, order.symbol, <date>)`. If `<date>` is `date.today()`, leave it (the rebalance runs same-day); if `submit_order` takes an `asof`/`dt` param, thread the rebalance `asof` through so catch-up runs on the same session reuse the same id. Add a focused test if the call site changes:

```python
# tests/execution/test_alpaca_coid.py (only if submit_order signature changes)
```

(If `submit_order` already calls `make_client_order_id(order.strategy_slug, order.symbol, date.today())`, no change is needed beyond Step 3 — note this in the commit.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/execution -v`
Expected: PASS (new deterministic tests pass; existing execution tests still green).

- [ ] **Step 6: Lint + types + commit**

```bash
uv run ruff format quant/execution/orders.py tests/execution/test_orders_deterministic.py
uv run ruff check quant/execution/orders.py && uv run mypy quant/execution/orders.py
git add quant/execution/orders.py tests/execution/test_orders_deterministic.py
git commit -m "fix(execution): deterministic client_order_id for broker idempotency"
```

---

## Task 10: Rebalance pre-submit marker + reconcile-then-refuse

**Files:**
- Modify: `quant/live/rebalance.py` (halt region ~180-194; submit loop 474-499)
- Test: `tests/live/test_rebalance_idempotency.py`

**Context:** A crash after Alpaca accepts orders but before the success marker would let the next tick re-submit. Two guards: (a) on entry, refuse if same-day orders already exist at the broker; (b) write a pre-submit marker the instant submission begins, before any commit step. Deterministic COID (Task 9) is the broker-level backstop.

- [ ] **Step 1: Write the failing test**

```python
"""Rebalance must refuse to submit if same-day orders already exist (idempotency)."""

from __future__ import annotations

from datetime import date

import pytest

from quant.live.rebalance import already_traded_today


class _ClientWithOrders:
    def list_orders_for_date(self, d: date) -> list[object]:
        return [object()]  # pretend one order already placed today


class _ClientNoOrders:
    def list_orders_for_date(self, d: date) -> list[object]:
        return []


def test_already_traded_today_true_when_orders_exist() -> None:
    assert already_traded_today(_ClientWithOrders(), date(2026, 6, 2)) is True


def test_already_traded_today_false_when_none() -> None:
    assert already_traded_today(_ClientNoOrders(), date(2026, 6, 2)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/live/test_rebalance_idempotency.py -v`
Expected: FAIL — `ImportError: cannot import name 'already_traded_today'`.

- [ ] **Step 3: Add the guard helper + wire it**

In `quant/live/rebalance.py`, add near the top-level helpers:

```python
def already_traded_today(client: object, asof: date) -> bool:
    """True iff the broker already has orders dated ``asof`` (idempotency guard).

    Uses a duck-typed ``list_orders_for_date(asof) -> list`` so tests can inject a
    fake. Defaults to False (fail-open to allow the trade) only if the client does
    not expose the method — but logs a warning, because without it the deterministic
    client_order_id (Task 9) is the sole double-submit backstop.
    """
    lister = getattr(client, "list_orders_for_date", None)
    if lister is None:
        logger.warning("client has no list_orders_for_date; relying on deterministic COID only")
        return False
    return len(lister(asof)) > 0
```

Then in `run_rebalance`, immediately AFTER the halt check (after line 194) and before the strategy loop, add:

```python
    if not dry_run and already_traded_today(client, asof):
        reason = f"orders already exist for {asof}; refusing to re-submit (idempotency)"
        logger.warning(reason)
        return RebalanceReport(
            asof=asof, equity=0.0, enabled_strategies=[], outcomes=[],
            dry_run=dry_run, safety_checks=safety_results, skipped_reason=reason,
        )
```

Add the pre-submit marker just BEFORE the net-submit loop (before line 478 `for order in net_orders(intended):`):

```python
    if not dry_run and intended and record_bookkeeping:
        from datetime import UTC, datetime

        from quant.deploy.markers import write_marker

        # Pre-submit marker: written the instant submission begins, BEFORE the
        # commit/push steps, so a post-submit crash blocks a re-fire next tick.
        write_marker(
            settings.data_dir, "daily-rebalance", asof,
            kind="SUBMITTED", fired_at_utc=datetime.now(UTC), exit_code=0, duration_s=0.0,
        )
```

> `list_orders_for_date` must exist on `AlpacaClient`. If it does not, add a thin method to `quant/execution/alpaca.py` that calls Alpaca's `get_orders(after=<asof 00:00>, until=<asof 23:59>)` and returns the list; add a `tests/execution` stub test. Keep it duck-typed so `run_rebalance` tests inject a fake.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/live/test_rebalance_idempotency.py tests/live -v`
Expected: PASS (new idempotency tests pass; existing rebalance tests green — they pass `dry_run=True` or inject clients).

- [ ] **Step 5: Lint + types + commit**

```bash
uv run ruff format quant/live/rebalance.py tests/live/test_rebalance_idempotency.py
uv run ruff check quant/live/rebalance.py && uv run mypy quant/live/rebalance.py
git add quant/live/rebalance.py tests/live/test_rebalance_idempotency.py
git commit -m "fix(live): reconcile-then-refuse + pre-submit marker (double-submit guard)"
```

---

## Task 11: Guard daemon self-ping

**Files:**
- Modify: `quant/monitor/daemon.py` (`run_loop` signature 189-202; tick body after line 233)
- Modify: `quant/cli.py` (`guard_run` `run_loop(...)` call, lines 1990-1999)
- Test: `tests/monitor/test_daemon_heartbeat.py`

- [ ] **Step 1: Write the failing test**

```python
"""The guard loop pings its OWN healthcheck each successful tick (decoupled
from the dispatcher), so guard-death is independently observable."""

from __future__ import annotations

from pathlib import Path

from quant.monitor.daemon import run_loop
from quant.monitor.guardrails import GuardrailConfig, GuardrailInputs


def test_run_loop_pings_healthcheck_each_tick(tmp_path: Path) -> None:
    pings: list[str] = []
    inputs = GuardrailInputs()  # empty inputs -> all guardrails OK
    run_loop(
        tmp_path,
        GuardrailConfig(),
        interval_s=0.0,
        max_ticks=2,
        inputs_fn=lambda: inputs,
        sleep=lambda _s: None,
        heartbeat_ping=lambda: pings.append("ok"),
    )
    assert pings == ["ok", "ok"]
```

> If `GuardrailInputs()`/`GuardrailConfig()` need required args, mirror an existing `tests/monitor/` test's construction (read that test and copy its fixture).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitor/test_daemon_heartbeat.py -v`
Expected: FAIL — `TypeError: run_loop() got an unexpected keyword argument 'heartbeat_ping'`.

- [ ] **Step 3: Add the `heartbeat_ping` seam**

In `quant/monitor/daemon.py`, add a parameter to `run_loop` (after `now_fn`, line 201):

```python
    heartbeat_ping: Callable[[], None] | None = None,
```

Inside the `try` block, immediately after `results.append(res)` (line 233), add:

```python
            if heartbeat_ping is not None:
                heartbeat_ping()  # best-effort liveness; inside try so a failure is caught
```

- [ ] **Step 4: Wire the CLI to ping the guard healthcheck**

In `quant/cli.py`, in `guard_run`, build an `AlertClient` and pass a ping closure. Replace the `run_loop(...)` call (lines 1990-1999) with:

```python
    from quant.deploy.alerts import AlertClient, AlertConfig

    _alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
        )
    )
    run_loop(
        settings.data_dir,
        config,
        interval_s=interval,
        dry_run=dry_run,
        max_ticks=max_ticks,
        alpaca_positions_fn=positions_fn,
        symbols=symbols,
        console_print=lambda s: console.print(s),
        heartbeat_ping=lambda: _alerts.ping_success(settings.healthcheck_guard_url),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/monitor/test_daemon_heartbeat.py tests/monitor -v`
Expected: PASS (new heartbeat test passes; existing monitor tests green).

- [ ] **Step 6: Lint + types + commit**

```bash
uv run ruff format quant/monitor/daemon.py quant/cli.py tests/monitor/test_daemon_heartbeat.py
uv run ruff check quant/monitor/daemon.py quant/cli.py && uv run mypy quant/monitor/daemon.py
git add quant/monitor/daemon.py quant/cli.py tests/monitor/test_daemon_heartbeat.py
git commit -m "feat(monitor): guard loop pings its own healthcheck (decoupled liveness)"
```

---

## Task 12: Dispatcher + `quant ops` CLI

**Files:**
- Create: `quant/deploy/dispatcher.py`
- Modify: `quant/cli.py` (add `ops` group near `guard`, ~line 1880)
- Test: `tests/deploy/test_dispatcher.py`

**Context:** The impure shell. Reads the real clock, loads manifest+markers, calls `due_jobs`, and for each Dispatch: runs the command chain (`uv run quant …` or expands `__script__`/`__backtest_matrix__`/`__validate_matrix__`), under a per-job non-blocking `flock`, writing a marker on success and pinging/alerting appropriately. All side-effecting collaborators are injected for tests.

- [ ] **Step 1: Write the failing test**

```python
"""Dispatcher tick: pure decision wired to injected runner/clock/alerts/lock."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.deploy.dispatcher import Dispatcher
from quant.deploy.manifest import load_manifest
from quant.deploy.markers import read_markers

MANIFEST = Path(__file__).resolve().parents[2] / "quant" / "deploy" / "jobs.toml"


class _Runner:
    def __init__(self, rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.rc = rc

    def __call__(self, args: list[str], cwd: Path) -> int:
        self.calls.append(args)
        return self.rc


class _Alerts:
    def __init__(self) -> None:
        self.success: list[str | None] = []
        self.emergencies: list[tuple[str, str]] = []

    def ping_success(self, url: str | None) -> None:
        self.success.append(url)

    def ping_fail(self, url: str | None, body: str = "") -> None:
        pass

    def send_emergency(self, title: str, message: str) -> bool:
        self.emergencies.append((title, message))
        return True


def _disp(tmp_path: Path, runner: _Runner, alerts: _Alerts) -> Dispatcher:
    return Dispatcher(
        data_dir=tmp_path,
        manifest=load_manifest(MANIFEST),
        runner=runner,
        alerts=alerts,
        halt_active=lambda: False,
    )


def test_fresh_premarket_runs_chain_and_marks(tmp_path: Path) -> None:
    runner, alerts = _Runner(), _Alerts()
    # Tue 2026-06-02 09:01 ET = 13:01 UTC (EDT)
    rc = _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 13, 1, tzinfo=UTC))
    assert rc == 0
    assert ["data", "refresh", "--start", "2018-01-01"] in runner.calls
    assert ["rebalance", "--dry-run"] in runner.calls
    assert read_markers(tmp_path).get("premarket-health") is not None
    assert alerts.success  # clean tick pinged liveness


def test_failed_chain_writes_no_marker_and_suppresses_success_ping(tmp_path: Path) -> None:
    runner, alerts = _Runner(rc=1), _Alerts()
    _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 13, 1, tzinfo=UTC))
    assert read_markers(tmp_path).get("premarket-health") is None  # no marker on failure
    assert alerts.success == []  # a failed job suppresses the success heartbeat


def test_doctor_failure_stops_rebalance_chain(tmp_path: Path) -> None:
    # A runner that fails ONLY on `doctor` must stop the daily-rebalance chain
    # before `rebalance` runs.
    class _DoctorFails(_Runner):
        def __call__(self, args: list[str], cwd: Path) -> int:
            self.calls.append(args)
            return 1 if args == ["doctor"] else 0

    runner, alerts = _DoctorFails(), _Alerts()
    # 2026-06-02 15:56 ET = 19:56 UTC -> rebalance FRESH window
    _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 19, 56, tzinfo=UTC))
    assert ["doctor"] in runner.calls
    assert ["rebalance"] not in runner.calls  # chain stopped at doctor


def test_missed_critical_fires_emergency_not_trade(tmp_path: Path) -> None:
    runner, alerts = _Runner(), _Alerts()
    # 2026-06-02 16:05 ET = 20:05 UTC -> past rebalance hard cutoff
    _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 20, 5, tzinfo=UTC))
    assert ["rebalance"] not in runner.calls
    assert any("rebalance" in m.lower() for _t, m in alerts.emergencies)


def test_halt_active_fires_emergency_and_runs_no_trade(tmp_path: Path) -> None:
    runner, alerts = _Runner(), _Alerts()
    d = Dispatcher(data_dir=tmp_path, manifest=load_manifest(MANIFEST), runner=runner,
                   alerts=alerts, halt_active=lambda: True)
    d.tick(now_utc=datetime(2026, 6, 2, 19, 56, tzinfo=UTC))
    assert ["rebalance"] not in runner.calls
    assert alerts.emergencies  # fresh halt pushed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_dispatcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant.deploy.dispatcher'`.

- [ ] **Step 3: Implement the dispatcher**

```python
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
    "defensive-etf-allocation", "trend", "momentum", "risk-parity", "multi-factor", "pairs",
)


class Alerts(Protocol):
    def ping_success(self, url: str | None) -> None: ...
    def ping_fail(self, url: str | None, body: str = "") -> None: ...
    def send_emergency(self, title: str, message: str) -> bool: ...


Runner = Callable[[list[str], Path], int]


def default_runner(args: list[str], cwd: Path) -> int:
    """Run `uv run quant <args>` (or a raw python script) and return the exit code."""
    cmd = [UV_BIN, "run", *args] if args[:1] != ["__python__"] else [UV_BIN, "run", "python", *args[1:]]
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


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
                out.append(["validate", slug, "--start", "2010-01-01",
                            "--bootstrap-resamples", "5000", "--bootstrap-seed", "0"])
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
            self.alerts.send_emergency("TRADING HALTED", f"halt active at {now_et:%Y-%m-%d %H:%M %Z}")

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
                write_marker(self.data_dir, disp.job.name, disp.session_date,
                             kind="MISSED_CRITICAL", fired_at_utc=now_utc, exit_code=-1, duration_s=0.0)
                continue
            if disp.kind == DispatchKind.MISSED:
                if self.alerts is not None:
                    self.alerts.ping_fail(self.tick_url, f"{disp.job.name} MISSED {disp.session_date}")
                write_marker(self.data_dir, disp.job.name, disp.session_date,
                             kind="MISSED", fired_at_utc=now_utc, exit_code=-1, duration_s=0.0)
                continue
            # FRESH or CATCH_UP -> run, under the job's lock (rebalance has its own).
            if self.halt_active() and disp.job.timing_critical:
                continue  # never trade while halted
            lock_name = "daily-rebalance" if disp.job.timing_critical else "batch"
            with _job_lock(self.data_dir, lock_name) as got:
                if not got:
                    logger.warning("lock {} busy; skipping {} this tick", lock_name, disp.job.name)
                    if disp.job.timing_critical and self.alerts is not None:
                        self.alerts.send_emergency("REBALANCE BLOCKED", f"{lock_name} lock held at trade window")
                    continue
                rc = self._run_chain(disp)
                if rc == 0:
                    write_marker(self.data_dir, disp.job.name, disp.session_date,
                                 kind=disp.kind, fired_at_utc=now_utc, exit_code=0, duration_s=0.0)
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
```

> Note: `data refresh` is intended non-fatal in the GH workflow. To preserve that, the manifest could mark per-step fatality; for the minimal plan, `doctor` (the blocking preflight) catches stale data, so a `data refresh` failure that still lets `doctor` pass is acceptable, and a `data refresh` failure that breaks `doctor` correctly stops the chain. If finer control is needed, add a `non_fatal_steps` field to `Job` in a follow-up.

- [ ] **Step 4: Add the `quant ops` CLI group**

In `quant/cli.py`, near the `guard` group (~line 1880), add:

```python
@cli.group(help="Deployment ops: the local tick scheduler (M4 host).")
def ops() -> None:
    pass


@ops.command("tick", help="One scheduler tick: run any due jobs. Called by launchd every 60s.")
def ops_tick() -> None:
    from quant.deploy.alerts import AlertClient, AlertConfig
    from quant.deploy.dispatcher import Dispatcher
    from quant.deploy.manifest import load_manifest
    from quant.governance.halt import load_halt

    settings = Settings()  # type: ignore[call-arg]
    alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
        )
    )
    manifest_path = Path(__file__).resolve().parent / "deploy" / "jobs.toml"
    disp = Dispatcher(
        data_dir=settings.data_dir,
        manifest=load_manifest(manifest_path),
        alerts=alerts,
        halt_active=lambda: load_halt(settings.data_dir).active,
        tick_url=settings.healthcheck_tick_url,
    )
    raise SystemExit(disp.tick())


@ops.command("run-job", help="Manually run one manifest job now (recovery for MISSED_CRITICAL).")
@click.argument("name")
@click.option("--force", is_flag=True, help="Run even if outside the window / already marked.")
def ops_run_job(name: str, force: bool) -> None:
    from quant.deploy.dispatcher import Dispatcher, _expand
    from quant.deploy.manifest import load_manifest

    settings = Settings()  # type: ignore[call-arg]
    manifest = load_manifest(Path(__file__).resolve().parent / "deploy" / "jobs.toml")
    job = next((j for j in manifest.jobs if j.name == name), None)
    if job is None:
        raise click.ClickException(f"unknown job: {name}")
    if not force:
        raise click.ClickException("refusing to run off-schedule without --force")
    disp = Dispatcher(data_dir=settings.data_dir, manifest=manifest)
    for args in _expand(job):
        rc = disp.runner(args, Path(__file__).resolve().parents[1])
        if rc != 0:
            raise SystemExit(rc)
```

> `Path` is already imported in cli.py; if not, add `from pathlib import Path`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/deploy/test_dispatcher.py -v && uv run quant ops --help`
Expected: PASS (5 passed); `quant ops` help lists `tick` and `run-job`.

- [ ] **Step 6: Lint + types + commit**

```bash
uv run ruff format quant/deploy/dispatcher.py quant/cli.py tests/deploy/test_dispatcher.py
uv run ruff check quant/deploy/dispatcher.py quant/cli.py && uv run mypy quant/deploy/dispatcher.py
git add quant/deploy/dispatcher.py quant/cli.py tests/deploy/test_dispatcher.py
git commit -m "feat(deploy): tick dispatcher + quant ops CLI (single-flight, halt-aware)"
```

---

## Task 13: launchd plists + installer + power + log rotation

**Files (all under `deploy/`):** `com.ajaiupadhyaya.quant-tick.plist`, `com.ajaiupadhyaya.quant-guard.plist`, `install.sh`, `uninstall.sh`, `pmset.sh`, `newsyslog/quant-deploy.conf`. These are OS-integration config — verified by shellcheck + `plutil -lint` + a dry `launchctl print`, not pytest.

- [ ] **Step 1: Write `deploy/com.ajaiupadhyaya.quant-tick.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.ajaiupadhyaya.quant-tick</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uv</string>
    <string>run</string>
    <string>quant</string>
    <string>ops</string>
    <string>tick</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/ajaiupadhyaya/Documents/quant-trading</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Standard</string>
  <key>StandardOutPath</key><string>/Users/ajaiupadhyaya/Library/Logs/quant-deploy/tick.stdout.log</string>
  <key>StandardErrorPath</key><string>/Users/ajaiupadhyaya/Library/Logs/quant-deploy/tick.stderr.log</string>
</dict>
</plist>
```

- [ ] **Step 2: Write `deploy/com.ajaiupadhyaya.quant-guard.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.ajaiupadhyaya.quant-guard</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uv</string>
    <string>run</string>
    <string>quant</string>
    <string>guard</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/ajaiupadhyaya/Documents/quant-trading</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Standard</string>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ExitTimeOut</key><integer>30</integer>
  <key>StandardOutPath</key><string>/Users/ajaiupadhyaya/Library/Logs/quant-deploy/guard.stdout.log</string>
  <key>StandardErrorPath</key><string>/Users/ajaiupadhyaya/Library/Logs/quant-deploy/guard.stderr.log</string>
</dict>
</plist>
```

> Bring-up note (in README): first load the guard with `--dry-run` by temporarily adding `<string>--dry-run</string>` after `run`, verify it cannot halt live, then remove and reload.

- [ ] **Step 3: Write `deploy/pmset.sh`**

```bash
#!/usr/bin/env bash
# Power settings for an always-on Mac mini. Persistent (survives reboot).
set -euo pipefail
sudo pmset -a sleep 0 disablesleep 1 displaysleep 0 disksleep 0 \
  autorestart 1 womp 1 powernap 0 standby 0 tcpkeepalive 1
echo "Applied. Verify with: pmset -g"
sudo systemsetup -setremotelogin on
```

- [ ] **Step 4: Write `deploy/install.sh` (idempotent: bootout then bootstrap)**

```bash
#!/usr/bin/env bash
set -euo pipefail
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/quant-deploy"
DOMAIN="gui/$(id -u)"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$AGENTS_DIR" "$LOG_DIR"
chmod 700 "$LOG_DIR"

for label in com.ajaiupadhyaya.quant-tick com.ajaiupadhyaya.quant-guard; do
  cp "$HERE/$label.plist" "$AGENTS_DIR/$label.plist"
  plutil -lint "$AGENTS_DIR/$label.plist"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$AGENTS_DIR/$label.plist"
  launchctl enable "$DOMAIN/$label"
  echo "loaded $label:"
  launchctl print "$DOMAIN/$label" | sed -n '1,5p'
done

# Log rotation (requires sudo; specify owner so the user-context agents keep write access).
sudo cp "$HERE/newsyslog/quant-deploy.conf" /etc/newsyslog.d/quant-deploy.conf
sudo newsyslog -nv /etc/newsyslog.d/quant-deploy.conf
echo "Install complete. Run ./pmset.sh separately (needs sudo)."
```

- [ ] **Step 5: Write `deploy/uninstall.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
DOMAIN="gui/$(id -u)"
for label in com.ajaiupadhyaya.quant-tick com.ajaiupadhyaya.quant-guard; do
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  rm -f "$HOME/Library/LaunchAgents/$label.plist"
done
sudo rm -f /etc/newsyslog.d/quant-deploy.conf
echo "Uninstalled launch agents + newsyslog config (logs + pmset left intact)."
```

- [ ] **Step 6: Write `deploy/newsyslog/quant-deploy.conf`**

```
# logfilename                                                      [owner:group]      mode count size when flags
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/tick.stdout.log     ajaiupadhyaya:staff 600  7     10000 *    GJ
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/tick.stderr.log     ajaiupadhyaya:staff 600  7     10000 *    GJ
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/guard.stdout.log    ajaiupadhyaya:staff 600  7     10000 *    GJ
/Users/ajaiupadhyaya/Library/Logs/quant-deploy/guard.stderr.log    ajaiupadhyaya:staff 600  7     10000 *    GJ
```

- [ ] **Step 7: Verify config validity (no pytest — these are OS files)**

```bash
chmod +x deploy/*.sh
plutil -lint deploy/com.ajaiupadhyaya.quant-tick.plist deploy/com.ajaiupadhyaya.quant-guard.plist
command -v shellcheck >/dev/null && shellcheck deploy/*.sh || echo "shellcheck not installed; skipping"
sudo newsyslog -nv deploy/newsyslog/quant-deploy.conf
```

Expected: `plutil` prints `OK` for both plists; shellcheck clean; `newsyslog -nv` prints the rotation plan without error.

- [ ] **Step 8: Commit**

```bash
git add deploy/com.ajaiupadhyaya.quant-tick.plist deploy/com.ajaiupadhyaya.quant-guard.plist \
  deploy/install.sh deploy/uninstall.sh deploy/pmset.sh deploy/newsyslog/quant-deploy.conf
git commit -m "feat(deploy): launchd plists + idempotent installer + pmset + newsyslog"
```

---

## Task 14: GitHub Actions migration + manifest fidelity test

**Files:**
- Modify: `.github/workflows/{premarket-health,daily-rebalance,posttrade-reconciliation,nightly-backtest,weekly-grid-search,weekly-validation-governance}.yml`
- Test: `tests/deploy/test_migration_fidelity.py`

- [ ] **Step 1: Write the failing test**

```python
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
    for name in ("daily-rebalance", "premarket-health", "posttrade-reconciliation",
                 "nightly-backtest", "weekly-grid-search", "weekly-validation-governance"):
        text = (wf / f"{name}.yml").read_text()
        assert "SCHEDULE RETIRED" in text, name
        # the active `schedule:` trigger is gone (only commented references remain)
        active = [ln for ln in text.splitlines() if ln.strip().startswith("schedule:")]
        assert active == [], f"{name} still has an active schedule: trigger"
        assert "workflow_dispatch" in text, name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/deploy/test_migration_fidelity.py -v`
Expected: FAIL — `test_retired_workflows_have_schedule_disabled` (workflows still have active `schedule:`).

- [ ] **Step 3: Edit each of the 6 workflow files**

For each file, (a) delete the `schedule:` block under `on:` (keep `workflow_dispatch:`), and (b) add a top-of-file comment. Example for `daily-rebalance.yml` — change the header to:

```yaml
# SCHEDULE RETIRED 2026-06 — executor migrated to the M4 tick scheduler.
# Source of truth for timing/commands: quant/deploy/jobs.toml.
# Kept for manual workflow_dispatch fallback + history.
name: daily-rebalance
on:
  workflow_dispatch:
```

Repeat for the other five (matching each file's existing `name:`). Leave the `jobs:` body unchanged so manual dispatch still works.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/deploy/test_migration_fidelity.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/*.yml tests/deploy/test_migration_fidelity.py
git commit -m "chore(ci): retire 6 cron schedules (M4 owns scheduling); keep manual dispatch"
```

---

## Task 15: Bring-up runbook + cutover gates

**Files:**
- Create: `deploy/README.md`

- [ ] **Step 1: Write `deploy/README.md`**

````markdown
# M4 Deployment (E1)

The M4 Mac mini is the sole always-on host. See the design spec:
`docs/superpowers/specs/2026-06-02-m4-deployment-e1-design.md`.

## One-time setup
1. `./deploy/pmset.sh`            # power: never sleep, auto-restart, WoL, remote login
2. Ensure FileVault is ON (System Settings → Privacy & Security). For planned
   reboots use: `sudo fdesetup authrestart`  (one unlock-free boot).
3. Fill `.env` (chmod 600) with PUSHOVER_APP_TOKEN, PUSHOVER_USER_KEY,
   HEALTHCHECKS_TICK_URL, HEALTHCHECKS_GUARD_URL.
4. Create 2 healthchecks.io checks (tick: period 1m grace 3m; guard: period 5m
   grace 11m) and wire each to the Pushover integration.
5. `mkdir -p ~/Library/Logs/quant-deploy`
6. Bring up the guard in dry-run first (edit the guard plist to add `--dry-run`),
   `./deploy/install.sh`, confirm it cannot halt live, then remove `--dry-run`
   and re-run install.

## Parallel dry-run (BEFORE retiring GH — but GH is already manual-only, so this
## is the M4 shakedown)
- Temporarily set the daily-rebalance manifest command to `["rebalance","--dry-run"]`.
- Run for several trading days. Confirm each day under `data/ops/scheduler/`:
  markers appear for premarket/rebalance/reconciliation; catch-up works after a
  forced sleep; the tick + guard healthchecks stay green; a deliberately-induced
  failure pages Pushover.

## Cutover gates (ALL must pass before live)
- [ ] Power-cycle survival: `sudo fdesetup authrestart`, confirm both agents
      reload and a "box recovered" state is observable after boot.
- [ ] Every alert channel tested: healthchecks "send test ping" → phone; a
      manual `quant` halt → emergency Pushover received & acked.
- [ ] A simulated MISSED_CRITICAL (tick after the close window) pages and does
      NOT trade.
- [ ] Dry-run outputs match the historical GH artifacts for the same days.
- [ ] Remove `--dry-run` from the rebalance manifest command; flip live.

## Operations
- Status: `launchctl print gui/$(id -u)/com.ajaiupadhyaya.quant-tick`
- Logs:   `~/Library/Logs/quant-deploy/{tick,guard}.{stdout,stderr}.log`
- Missed rebalance recovery (if still pre-close): `uv run quant ops run-job daily-rebalance --force`
- Resume after a halt (human only): `uv run quant governance resume --reason "..."`
- Uninstall: `./deploy/uninstall.sh`

## Accepted risks (M4-only + FileVault Path B)
- No off-box executor fallback; an M4 outage in the ~3-min close window = a
  missed rebalance handled manually (low-harm: daily defensive-ETF book).
- Unexpected power loss → box at FileVault unlock screen; dead-man's-switch
  pages you; recover via SSH/physical unlock.
- healthchecks.io is a 3rd-party SPOF; a weekly synthetic test push you must ack
  detects silent push death.
````

- [ ] **Step 2: Commit**

```bash
git add deploy/README.md
git commit -m "docs(deploy): M4 bring-up runbook + cutover gates"
```

---

## Final verification

- [ ] **Full suite + lint + types green**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy quant && uv run pytest -m "not alpaca and not network and not slow"
```
Expected: ruff clean, format clean, mypy `Success`, pytest all passed (no regressions; ~30+ new deploy tests pass).

- [ ] **CLI smoke**

```bash
uv run quant ops --help && uv run quant ops tick   # tick is a no-op outside any job window on a non-trading moment
```
Expected: help lists `tick`/`run-job`; a tick outside all windows exits 0 having run nothing.

---

## Self-review notes (addressed in this plan)

- **Spec coverage:** every E1 spec section maps to a task — calendar_clock/manifest/scheduler/markers (Tasks 4-7), dispatcher+CLI (12), alerts (8), trade idempotency (9-10), halt/status durability (3), guard self-ping (11), launchd/power/logs (13), GH migration (14), runbook/gates (15), config (2), atomic util (1).
- **Type consistency:** shared-types block is the contract; `Dispatch`, `Job`, `DispatchKind`, `AlertConfig`, `make_client_order_id`, `write_marker` signatures are identical across Tasks 5-12.
- **Deferred to follow-ups (out of E1 scope, noted in spec §15):** per-step `non_fatal` flags, NTP/disk-floor guards as code (documented as host checks in the runbook), `data/` retention policy, deterministic-COID change to `alpaca.py` call site is verified-not-rewritten in Task 9 Step 4.
````
