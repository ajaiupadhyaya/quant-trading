# Autonomous Monitoring Daemon Implementation Plan (Pillar 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless guardian daemon (`quant/monitor/`) that evaluates guardrails (drift, account drawdown, reconciliation, bar freshness) each tick and auto-pulls the existing kill-switch (`set_halt`) on a halt verdict, streaming a terminal heartbeat and writing a status artifact. Plus a `quant guard` CLI (`check`, `run`).

**Architecture:** Pure guardrail layer (`guardrails.py`) → status artifact (`status.py`) → I/O orchestration (`daemon.py`: `gather_inputs`/`run_once`/`run_loop`) → CLI (`quant guard`). The daemon can HALT but NEVER resumes (resume is always manual). Its only writes are `set_halt` + the status JSON; it never touches orders. Fail-safe: monitoring failures degrade to "ok"/"warn" and never crash the loop or trigger a false halt.

**Tech Stack:** Python 3.12, pandas, Click, rich, pytest. uv-managed. mypy-strict, ruff lint+format. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-monitoring-daemon-design.md`

**Tooling note:** Use `uv run <cmd>` for ALL commands (`uv run pytest`, `uv run ruff check .`, `uv run ruff format .`, `uv run mypy quant`, `uv run quant ...`). Never `.venv/bin/...`, never bare `python`/`pip`/`pytest`.

---

## File Structure

- Create `quant/monitor/__init__.py` — public exports (filled in Task 5).
- Create `quant/monitor/guardrails.py` — pure guardrail layer (Task 1).
- Create `quant/monitor/status.py` — status artifact (Task 2).
- Create `quant/monitor/daemon.py` — `gather_inputs`, `run_once`, `format_heartbeat`, `TickResult` (Task 3); `run_loop` (Task 4).
- Modify `quant/cli.py` — add `guard` group + `check`/`run` (Task 6).
- Create `tests/monitor/__init__.py` + test modules.
- Modify `README.md` (Task 7).

---

### Task 1: Pure guardrail layer

**Files:**
- Create: `quant/monitor/__init__.py` (docstring only for now)
- Create: `quant/monitor/guardrails.py`
- Test: `tests/monitor/__init__.py`, `tests/monitor/test_guardrails.py`

- [ ] **Step 1: Write failing tests**

Create `tests/monitor/__init__.py` (empty). Create `tests/monitor/test_guardrails.py`:

```python
from __future__ import annotations

from quant.governance.drift import DriftRow
from quant.live.safety import CheckResult, StrategyRiskBudget
from quant.monitor.guardrails import (
    GuardrailConfig,
    GuardrailInputs,
    evaluate_account_drawdown,
    evaluate_bar_freshness,
    evaluate_drift,
    evaluate_guardrails,
    evaluate_reconciliation,
)


def _row(flag: str, window: int = 20, z: float = -2.5) -> DriftRow:
    return DriftRow(
        strategy="account",
        window=window,
        realized_return=-0.1,
        expected_return=0.0,
        z_score=z,
        flag=flag,  # type: ignore[arg-type]
    )


def test_drift_halt_on_halt_candidate() -> None:
    out = evaluate_drift([_row("normal"), _row("halt_candidate")])
    assert out.severity == "halt"
    assert out.name == "drift"


def test_drift_warn_on_watch_only() -> None:
    out = evaluate_drift([_row("normal"), _row("watch")])
    assert out.severity == "warn"


def test_drift_ok_when_normal_or_empty() -> None:
    assert evaluate_drift([_row("normal")]).severity == "ok"
    assert evaluate_drift([]).severity == "ok"


def test_account_drawdown_halt_on_breach() -> None:
    budget = StrategyRiskBudget(max_drawdown=0.25)
    assert evaluate_account_drawdown(-0.30, budget).severity == "halt"
    assert evaluate_account_drawdown(-0.25, budget).severity == "halt"  # at threshold
    assert evaluate_account_drawdown(-0.10, budget).severity == "ok"
    assert evaluate_account_drawdown(0.0, budget).severity == "ok"


def test_reconciliation_severity() -> None:
    assert evaluate_reconciliation(None, halt_on_breach=False).severity == "ok"
    ok = CheckResult(ok=True, name="reconciliation", detail="fine")
    assert evaluate_reconciliation(ok, halt_on_breach=False).severity == "ok"
    bad = CheckResult(ok=False, name="reconciliation", detail="diff")
    assert evaluate_reconciliation(bad, halt_on_breach=False).severity == "warn"
    assert evaluate_reconciliation(bad, halt_on_breach=True).severity == "halt"


def test_bar_freshness_severity() -> None:
    assert evaluate_bar_freshness(None).severity == "ok"
    ok = CheckResult(ok=True, name="bar_freshness", detail="fresh")
    assert evaluate_bar_freshness(ok).severity == "ok"
    stale = CheckResult(ok=False, name="bar_freshness", detail="stale")
    assert evaluate_bar_freshness(stale).severity == "warn"


def test_evaluate_guardrails_aggregates_worst() -> None:
    inputs = GuardrailInputs(
        drift_rows=[_row("halt_candidate")],
        account_drawdown_pct=-0.05,
        latest_equity=100_000.0,
        reconciliation=CheckResult(ok=True, name="reconciliation", detail=""),
        bar_freshness=CheckResult(ok=False, name="bar_freshness", detail="stale"),
    )
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert {o.name for o in report.outcomes} == {
        "drift",
        "account_drawdown",
        "reconciliation",
        "bar_freshness",
    }
    assert report.worst_severity == "halt"
    assert report.halting is True


def test_evaluate_guardrails_all_ok() -> None:
    inputs = GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=0.0,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert report.worst_severity == "ok"
    assert report.halting is False
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/monitor/test_guardrails.py -q`
Expected: FAIL — `ModuleNotFoundError: quant.monitor.guardrails`.

- [ ] **Step 3: Implement**

Create `quant/monitor/__init__.py`:

```python
"""Autonomous monitoring daemon + kill-switch — the headless guardian (pillar 2)."""
```

Create `quant/monitor/guardrails.py`:

```python
"""Pure guardrail evaluation. No I/O, no side effects, total functions.

Each guardrail inspects one aspect of book health and yields a
``GuardrailOutcome`` with severity in {ok, warn, halt}. The overall tick halts
iff any guardrail returns ``halt``. Halt authority belongs to drift and
account-drawdown (computed from authoritative local equity history);
reconciliation and bar-freshness are warn-only by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from quant.governance.drift import DriftConfig, DriftRow
from quant.live.safety import CheckResult, StrategyRiskBudget

Severity = Literal["ok", "warn", "halt"]
_RANK: dict[Severity, int] = {"ok": 0, "warn": 1, "halt": 2}


@dataclass(frozen=True)
class GuardrailOutcome:
    name: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class GuardrailConfig:
    drift: DriftConfig = field(default_factory=DriftConfig)
    risk: StrategyRiskBudget = field(default_factory=StrategyRiskBudget)
    reconciliation_is_halt: bool = False


@dataclass(frozen=True)
class GuardrailInputs:
    drift_rows: list[DriftRow]
    account_drawdown_pct: float  # non-positive
    latest_equity: float
    reconciliation: CheckResult | None  # None => skipped (no live account)
    bar_freshness: CheckResult | None  # None => skipped


@dataclass(frozen=True)
class GuardrailReport:
    outcomes: list[GuardrailOutcome]

    @property
    def worst_severity(self) -> Severity:
        worst: Severity = "ok"
        for o in self.outcomes:
            if _RANK[o.severity] > _RANK[worst]:
                worst = o.severity
        return worst

    @property
    def halting(self) -> bool:
        return self.worst_severity == "halt"


def evaluate_drift(rows: list[DriftRow]) -> GuardrailOutcome:
    halts = [r for r in rows if r.flag == "halt_candidate"]
    if halts:
        which = ", ".join(f"{r.strategy}@{r.window}d z={r.z_score:.2f}" for r in halts[:5])
        return GuardrailOutcome("drift", "halt", f"halt_candidate: {which}")
    watches = [r for r in rows if r.flag == "watch"]
    if watches:
        which = ", ".join(f"{r.strategy}@{r.window}d z={r.z_score:.2f}" for r in watches[:5])
        return GuardrailOutcome("drift", "warn", f"watch: {which}")
    if not rows:
        return GuardrailOutcome("drift", "ok", "no drift history")
    return GuardrailOutcome("drift", "ok", "all windows normal")


def evaluate_account_drawdown(dd_pct: float, budget: StrategyRiskBudget) -> GuardrailOutcome:
    cap = abs(budget.max_drawdown)
    if dd_pct <= -cap:
        return GuardrailOutcome(
            "account_drawdown", "halt", f"drawdown {dd_pct:.2%} <= -{cap:.2%}"
        )
    return GuardrailOutcome(
        "account_drawdown", "ok", f"drawdown {dd_pct:.2%} within -{cap:.2%}"
    )


def evaluate_reconciliation(
    recon: CheckResult | None, *, halt_on_breach: bool
) -> GuardrailOutcome:
    if recon is None:
        return GuardrailOutcome("reconciliation", "ok", "skipped: no account")
    if recon.ok:
        return GuardrailOutcome("reconciliation", "ok", recon.detail)
    severity: Severity = "halt" if halt_on_breach else "warn"
    return GuardrailOutcome("reconciliation", severity, recon.detail)


def evaluate_bar_freshness(freshness: CheckResult | None) -> GuardrailOutcome:
    if freshness is None:
        return GuardrailOutcome("bar_freshness", "ok", "skipped")
    if freshness.ok:
        return GuardrailOutcome("bar_freshness", "ok", freshness.detail)
    return GuardrailOutcome("bar_freshness", "warn", freshness.detail)


def evaluate_guardrails(inputs: GuardrailInputs, config: GuardrailConfig) -> GuardrailReport:
    outcomes = [
        evaluate_drift(inputs.drift_rows),
        evaluate_account_drawdown(inputs.account_drawdown_pct, config.risk),
        evaluate_reconciliation(
            inputs.reconciliation, halt_on_breach=config.reconciliation_is_halt
        ),
        evaluate_bar_freshness(inputs.bar_freshness),
    ]
    return GuardrailReport(outcomes=outcomes)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/monitor/test_guardrails.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/monitor tests/monitor && uv run ruff format quant/monitor tests/monitor && uv run mypy quant/monitor`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/monitor/__init__.py quant/monitor/guardrails.py tests/monitor/__init__.py tests/monitor/test_guardrails.py
git commit -m "feat(monitor): pure guardrail evaluation layer"
```

---

### Task 2: Status artifact

**Files:**
- Create: `quant/monitor/status.py`
- Test: `tests/monitor/test_status.py`

- [ ] **Step 1: Write failing tests**

Create `tests/monitor/test_status.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.monitor.guardrails import GuardrailOutcome
from quant.monitor.status import (
    MonitorStatus,
    monitor_status_path,
    read_status,
    write_status,
)


def test_status_path(tmp_path: Path) -> None:
    assert monitor_status_path(tmp_path) == tmp_path / "ops" / "monitor_status.json"


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    status = MonitorStatus(
        version=1,
        at=datetime(2026, 5, 28, 14, 32, 5, tzinfo=UTC),
        worst_severity="halt",
        halt_triggered_this_tick=True,
        halt_active=True,
        outcomes=[
            GuardrailOutcome("drift", "halt", "halt_candidate: account@20d z=-2.50"),
            GuardrailOutcome("account_drawdown", "ok", "drawdown -1.00% within -25.00%"),
        ],
        heartbeat="14:32:05 | equity $100,000 dd -1.0% | drift halt | ...",
    )
    write_status(tmp_path, status)
    loaded = read_status(tmp_path)
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.at == status.at
    assert loaded.worst_severity == "halt"
    assert loaded.halt_triggered_this_tick is True
    assert loaded.halt_active is True
    assert [(o.name, o.severity, o.detail) for o in loaded.outcomes] == [
        (o.name, o.severity, o.detail) for o in status.outcomes
    ]
    assert loaded.heartbeat == status.heartbeat


def test_read_status_absent_returns_none(tmp_path: Path) -> None:
    assert read_status(tmp_path) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/monitor/test_status.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `quant/monitor/status.py`:

```python
"""On-disk status artifact for the monitoring daemon: data/ops/monitor_status.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant.monitor.guardrails import GuardrailOutcome, Severity


@dataclass(frozen=True)
class MonitorStatus:
    version: int
    at: datetime
    worst_severity: Severity
    halt_triggered_this_tick: bool
    halt_active: bool
    outcomes: list[GuardrailOutcome]
    heartbeat: str


def monitor_status_path(data_dir: Path) -> Path:
    return data_dir / "ops" / "monitor_status.json"


def write_status(data_dir: Path, status: MonitorStatus) -> Path:
    path = monitor_status_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": status.version,
        "at": status.at.isoformat(),
        "worst_severity": status.worst_severity,
        "halt_triggered_this_tick": status.halt_triggered_this_tick,
        "halt_active": status.halt_active,
        "outcomes": [
            {"name": o.name, "severity": o.severity, "detail": o.detail}
            for o in status.outcomes
        ],
        "heartbeat": status.heartbeat,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_status(data_dir: Path) -> MonitorStatus | None:
    path = monitor_status_path(data_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    outcomes = [
        GuardrailOutcome(name=str(o["name"]), severity=o["severity"], detail=str(o["detail"]))
        for o in raw.get("outcomes", [])
    ]
    return MonitorStatus(
        version=int(raw["version"]),
        at=datetime.fromisoformat(str(raw["at"])),
        worst_severity=raw["worst_severity"],
        halt_triggered_this_tick=bool(raw["halt_triggered_this_tick"]),
        halt_active=bool(raw["halt_active"]),
        outcomes=outcomes,
        heartbeat=str(raw["heartbeat"]),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/monitor/test_status.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/monitor tests/monitor && uv run ruff format quant/monitor tests/monitor && uv run mypy quant/monitor`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/monitor/status.py tests/monitor/test_status.py
git commit -m "feat(monitor): monitor_status.json artifact (write/read)"
```

---

### Task 3: Daemon — gather_inputs, run_once, heartbeat

**Files:**
- Create: `quant/monitor/daemon.py`
- Test: `tests/monitor/test_daemon.py`

- [ ] **Step 1: Write failing tests**

Create `tests/monitor/test_daemon.py`:

```python
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from quant.governance.drift import DriftRow
from quant.governance.halt import load_halt, set_halt
from quant.live.safety import CheckResult
from quant.monitor.daemon import format_heartbeat, gather_inputs, run_once
from quant.monitor.guardrails import GuardrailConfig, GuardrailInputs, evaluate_guardrails
from quant.monitor.status import read_status

NOW = datetime(2026, 5, 28, 14, 0, 0, tzinfo=UTC)


def _halt_row() -> DriftRow:
    return DriftRow(
        strategy="account",
        window=20,
        realized_return=-0.2,
        expected_return=0.0,
        z_score=-3.0,
        flag="halt_candidate",
    )


def _clean_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.02,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def _drawdown_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.30,
        latest_equity=70_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def test_run_once_clean_no_halt(tmp_path: Path) -> None:
    res = run_once(tmp_path, GuardrailConfig(), inputs=_clean_inputs(), now=NOW)
    assert res.halt_triggered is False
    assert res.halt_active is False
    assert res.report.worst_severity == "ok"
    assert load_halt(tmp_path).active is False
    status = read_status(tmp_path)
    assert status is not None and status.worst_severity == "ok"


def test_run_once_drawdown_breach_halts(tmp_path: Path) -> None:
    res = run_once(tmp_path, GuardrailConfig(), inputs=_drawdown_inputs(), now=NOW)
    assert res.halt_triggered is True
    assert res.halt_active is True
    halt = load_halt(tmp_path)
    assert halt.active is True
    assert "account_drawdown" in halt.reason


def test_run_once_drift_halt_candidate_halts(tmp_path: Path) -> None:
    inputs = GuardrailInputs(
        drift_rows=[_halt_row()],
        account_drawdown_pct=-0.01,
        latest_equity=99_000.0,
        reconciliation=None,
        bar_freshness=None,
    )
    res = run_once(tmp_path, GuardrailConfig(), inputs=inputs, now=NOW)
    assert res.halt_triggered is True
    assert "drift" in load_halt(tmp_path).reason


def test_run_once_idempotent_when_already_halted(tmp_path: Path) -> None:
    set_halt(tmp_path, reason="manual: original reason", created_at=NOW)
    res = run_once(tmp_path, GuardrailConfig(), inputs=_drawdown_inputs(), now=NOW)
    assert res.halt_triggered is False  # did not re-halt
    assert res.halt_active is True
    # original reason preserved
    assert load_halt(tmp_path).reason == "manual: original reason"


def test_run_once_dry_run_does_not_halt(tmp_path: Path) -> None:
    res = run_once(tmp_path, GuardrailConfig(), inputs=_drawdown_inputs(), now=NOW, dry_run=True)
    assert res.halt_triggered is False
    assert load_halt(tmp_path).active is False  # NOT halted
    status = read_status(tmp_path)
    assert status is not None and status.worst_severity == "halt"  # but recorded


def test_run_once_warn_only_does_not_halt(tmp_path: Path) -> None:
    inputs = GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.05,
        latest_equity=100_000.0,
        reconciliation=CheckResult(ok=False, name="reconciliation", detail="diff"),
        bar_freshness=CheckResult(ok=False, name="bar_freshness", detail="stale"),
    )
    res = run_once(tmp_path, GuardrailConfig(), inputs=inputs, now=NOW)
    assert res.report.worst_severity == "warn"
    assert res.halt_triggered is False
    assert load_halt(tmp_path).active is False


def test_gather_inputs_missing_equity_is_failsafe(tmp_path: Path) -> None:
    # No equity.parquet -> all-ok inputs, never a false halt.
    inputs = gather_inputs(tmp_path, asof=date(2026, 5, 28), config=GuardrailConfig())
    assert inputs.drift_rows == []
    assert inputs.account_drawdown_pct == 0.0
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert report.halting is False


def test_gather_inputs_reads_equity_and_computes_drawdown(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir(parents=True)
    # ramp to 100 then crash to 70 -> ~ -30% drawdown
    equity = [90.0, 95.0, 100.0, 100.0, 100.0, 100.0, 85.0, 70.0]
    df = pd.DataFrame(
        {"date": pd.bdate_range("2026-01-01", periods=len(equity)), "equity": equity}
    )
    df.to_parquet(live / "equity.parquet")
    inputs = gather_inputs(tmp_path, asof=date(2026, 5, 28), config=GuardrailConfig())
    assert inputs.account_drawdown_pct <= -0.29
    assert inputs.latest_equity == 70.0


def test_format_heartbeat_marks_halt() -> None:
    inputs = _drawdown_inputs()
    report = evaluate_guardrails(inputs, GuardrailConfig())
    line = format_heartbeat(inputs, report, NOW, halt_active=True)
    assert "HALT" in line
    assert "dd" in line
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/monitor/test_daemon.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `quant/monitor/daemon.py` (Task 4 will append `run_loop`):

```python
"""Monitoring daemon orchestration: gather inputs, run a tick, write status.

The daemon can HALT but NEVER resumes — resume is always a manual
`quant governance resume`. Its only side effects are ``set_halt`` (the
kill-switch) and the status artifact; it never touches orders. Fail-safe:
on missing/empty equity the inputs evaluate to ``ok`` so a monitoring gap
cannot trigger a false halt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from quant.governance.drift import DriftRow, summarize_drift
from quant.governance.halt import load_halt, set_halt
from quant.live.bookkeeping import read_equity
from quant.live.safety import (
    CheckResult,
    check_bar_freshness,
    check_reconciliation,
    enabled_strategy_slugs,
)
from quant.execution.alpaca import PositionRow
from quant.monitor.guardrails import (
    GuardrailConfig,
    GuardrailInputs,
    GuardrailReport,
    evaluate_guardrails,
)
from quant.monitor.status import MonitorStatus, write_status


@dataclass(frozen=True)
class TickResult:
    report: GuardrailReport
    halt_triggered: bool  # set_halt was called THIS tick
    halt_active: bool  # halt active after this tick
    heartbeat: str
    at: datetime


def _drift_rows(equity_df: pd.DataFrame, config: GuardrailConfig) -> list[DriftRow]:
    if equity_df.empty or "equity" not in equity_df.columns:
        return []
    returns = equity_df["equity"].astype(float).pct_change(fill_method=None).dropna()
    if returns.empty:
        return []
    realized = {"account": returns}
    expected = {"account": pd.Series(0.0, index=returns.index)}
    return summarize_drift(realized, expected, config=config.drift)


def _account_drawdown(equity_df: pd.DataFrame, lookback_days: int) -> float:
    """Worst trailing peak-to-trough drawdown. Mirrors safety._recent_drawdown_pct."""
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    window = equity_df.tail(lookback_days)
    if window.empty:
        return 0.0
    equity = window["equity"].astype(float)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _latest_equity(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    return float(equity_df["equity"].astype(float).iloc[-1])


def gather_inputs(
    data_dir,  # type: ignore[no-untyped-def]
    *,
    asof: date,
    config: GuardrailConfig,
    alpaca_positions: list[PositionRow] | None = None,
    symbols: list[str] | None = None,
) -> GuardrailInputs:
    """Read local state into a GuardrailInputs. I/O side of the daemon.

    Reconciliation is included only when ``alpaca_positions`` is provided;
    bar-freshness only when ``symbols`` is non-empty. Both default to skipped.
    """
    equity_df = read_equity(data_dir)
    drift_rows = _drift_rows(equity_df, config)
    dd = _account_drawdown(equity_df, config.risk.drawdown_lookback_days)

    reconciliation: CheckResult | None = None
    if alpaca_positions is not None:
        reconciliation = check_reconciliation(
            data_dir=data_dir,
            alpaca_positions=alpaca_positions,
            enabled_slugs=enabled_strategy_slugs(),
        )

    freshness: CheckResult | None = None
    if symbols:
        freshness = check_bar_freshness(data_dir, symbols=symbols, asof=asof)

    return GuardrailInputs(
        drift_rows=drift_rows,
        account_drawdown_pct=dd,
        latest_equity=_latest_equity(equity_df),
        reconciliation=reconciliation,
        bar_freshness=freshness,
    )


def format_heartbeat(
    inputs: GuardrailInputs,
    report: GuardrailReport,
    now: datetime,
    *,
    halt_active: bool,
) -> str:
    ts = now.strftime("%H:%M:%S")
    eq = f"${inputs.latest_equity:,.0f}"
    dd = f"{inputs.account_drawdown_pct:.1%}"
    parts = " | ".join(f"{o.name} {o.severity}" for o in report.outcomes)
    line = f"{ts} | equity {eq} dd {dd} | {parts}"
    if halt_active:
        line += " | [HALT]"
    return line


def run_once(
    data_dir,  # type: ignore[no-untyped-def]
    config: GuardrailConfig,
    *,
    asof: date | None = None,
    inputs: GuardrailInputs | None = None,
    alpaca_positions: list[PositionRow] | None = None,
    symbols: list[str] | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> TickResult:
    """One monitoring tick. Evaluate guardrails; auto-halt on a halt verdict
    (unless already halted or dry_run); always write the status artifact."""
    asof = asof or date.today()
    now = now or datetime.now(UTC).replace(microsecond=0)
    if inputs is None:
        inputs = gather_inputs(
            data_dir,
            asof=asof,
            config=config,
            alpaca_positions=alpaca_positions,
            symbols=symbols,
        )

    report = evaluate_guardrails(inputs, config)
    halt_active = load_halt(data_dir).active
    triggered = False
    if report.halting and not halt_active and not dry_run:
        names = ",".join(o.name for o in report.outcomes if o.severity == "halt")
        set_halt(data_dir, reason=f"auto-halt: {names}", created_at=now)
        halt_active = True
        triggered = True

    heartbeat = format_heartbeat(inputs, report, now, halt_active=halt_active)
    write_status(
        data_dir,
        MonitorStatus(
            version=1,
            at=now,
            worst_severity=report.worst_severity,
            halt_triggered_this_tick=triggered,
            halt_active=halt_active,
            outcomes=report.outcomes,
            heartbeat=heartbeat,
        ),
    )
    return TickResult(
        report=report,
        halt_triggered=triggered,
        halt_active=halt_active,
        heartbeat=heartbeat,
        at=now,
    )
```

Note: `data_dir` is annotated loosely (`# type: ignore[no-untyped-def]`) to avoid importing `Path` typing friction with the existing call sites — actually DO type it properly: import `from pathlib import Path` and annotate `data_dir: Path`. Replace the `# type: ignore[no-untyped-def]` markers accordingly. (The marker is only a fallback if a typing conflict appears; prefer the explicit `Path` annotation and no ignore.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/monitor/test_daemon.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/monitor tests/monitor && uv run ruff format quant/monitor tests/monitor && uv run mypy quant/monitor`
Expected: clean. (Ensure `data_dir: Path` is typed; remove stray ignores if mypy is happy.)

- [ ] **Step 6: Commit**

```bash
git add quant/monitor/daemon.py tests/monitor/test_daemon.py
git commit -m "feat(monitor): daemon tick — gather inputs, evaluate, auto-halt, status"
```

---

### Task 4: Daemon — run_loop

**Files:**
- Modify: `quant/monitor/daemon.py`
- Test: `tests/monitor/test_loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/monitor/test_loop.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.governance.halt import load_halt
from quant.monitor.daemon import run_loop
from quant.monitor.guardrails import GuardrailConfig, GuardrailInputs

NOW = datetime(2026, 5, 28, 14, 0, 0, tzinfo=UTC)


def _clean_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.01,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def test_run_loop_runs_max_ticks_without_real_sleep(tmp_path: Path) -> None:
    slept: list[float] = []
    printed: list[str] = []
    results = run_loop(
        tmp_path,
        GuardrailConfig(),
        interval_s=999.0,
        max_ticks=3,
        inputs_fn=lambda: _clean_inputs(),
        sleep=lambda s: slept.append(s),
        console_print=printed.append,
        now_fn=lambda: NOW,
    )
    assert len(results) == 3
    assert len(printed) == 3
    # sleeps only happen BETWEEN ticks, not after the last
    assert len(slept) == 2
    assert load_halt(tmp_path).active is False


def test_run_loop_continues_on_tick_error(tmp_path: Path) -> None:
    printed: list[str] = []
    calls = {"n": 0}

    def flaky_inputs() -> GuardrailInputs:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _clean_inputs()

    results = run_loop(
        tmp_path,
        GuardrailConfig(),
        interval_s=0.0,
        max_ticks=2,
        inputs_fn=flaky_inputs,
        sleep=lambda s: None,
        console_print=printed.append,
        now_fn=lambda: NOW,
    )
    # first tick errored (no TickResult), second succeeded
    assert len(results) == 1
    assert any("error" in line.lower() for line in printed)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/monitor/test_loop.py -q`
Expected: FAIL — `run_loop` not defined / no `inputs_fn` param.

- [ ] **Step 3: Implement — append to `quant/monitor/daemon.py`**

Add these imports at the top of `daemon.py` (merge with existing import block):

```python
import time
from collections.abc import Callable
```

Append at the end of `daemon.py`:

```python
def run_loop(
    data_dir: Path,
    config: GuardrailConfig,
    *,
    interval_s: float = 300.0,
    dry_run: bool = False,
    max_ticks: int | None = None,
    inputs_fn: Callable[[], GuardrailInputs] | None = None,
    alpaca_positions_fn: Callable[[], list[PositionRow] | None] | None = None,
    symbols: list[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    console_print: Callable[[str], None] | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> list[TickResult]:
    """Repeatedly run a tick, printing the heartbeat, sleeping between ticks.

    Fail-safe: a tick that raises is caught, reported via ``console_print``, and
    the loop continues. ``inputs_fn`` (test seam) supplies inputs directly;
    otherwise ``alpaca_positions_fn`` is consulted each tick for reconciliation.
    Stops after ``max_ticks`` ticks (None = forever).
    """
    results: list[TickResult] = []
    tick = 0
    while max_ticks is None or tick < max_ticks:
        try:
            tick_inputs = inputs_fn() if inputs_fn is not None else None
            positions = (
                alpaca_positions_fn()
                if (tick_inputs is None and alpaca_positions_fn is not None)
                else None
            )
            now = now_fn() if now_fn is not None else None
            res = run_once(
                data_dir,
                config,
                inputs=tick_inputs,
                alpaca_positions=positions,
                symbols=symbols,
                dry_run=dry_run,
                now=now,
            )
            if console_print is not None:
                console_print(res.heartbeat)
            results.append(res)
        except Exception as exc:  # noqa: BLE001 — fail-safe: never crash the loop
            if console_print is not None:
                console_print(f"monitor tick error (continuing): {exc!r}")
        tick += 1
        if max_ticks is not None and tick >= max_ticks:
            break
        sleep(interval_s)
    return results
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/monitor/test_loop.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/monitor tests/monitor && uv run ruff format quant/monitor tests/monitor && uv run mypy quant/monitor`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/monitor/daemon.py tests/monitor/test_loop.py
git commit -m "feat(monitor): fail-safe run_loop with injectable sleep/print/inputs"
```

---

### Task 5: Public exports

**Files:**
- Modify: `quant/monitor/__init__.py`
- Test: `tests/monitor/test_exports.py`

- [ ] **Step 1: Write failing test**

Create `tests/monitor/test_exports.py`:

```python
from __future__ import annotations

from quant.monitor import (
    GuardrailConfig,
    GuardrailInputs,
    GuardrailOutcome,
    GuardrailReport,
    MonitorStatus,
    TickResult,
    evaluate_guardrails,
    gather_inputs,
    monitor_status_path,
    read_status,
    run_loop,
    run_once,
)


def test_public_api_importable() -> None:
    assert GuardrailConfig and GuardrailInputs and GuardrailOutcome and GuardrailReport
    assert MonitorStatus and TickResult
    assert evaluate_guardrails and gather_inputs and run_once and run_loop
    assert monitor_status_path and read_status
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/monitor/test_exports.py -q`
Expected: FAIL — names not exported.

- [ ] **Step 3: Implement — overwrite `quant/monitor/__init__.py`**

```python
"""Autonomous monitoring daemon + kill-switch — the headless guardian (pillar 2).

The daemon evaluates guardrails each tick and auto-pulls the existing
kill-switch on a halt verdict. It can HALT but never resumes (resume is always
a manual `quant governance resume`).
"""

from quant.monitor.daemon import (
    TickResult,
    format_heartbeat,
    gather_inputs,
    run_loop,
    run_once,
)
from quant.monitor.guardrails import (
    GuardrailConfig,
    GuardrailInputs,
    GuardrailOutcome,
    GuardrailReport,
    Severity,
    evaluate_account_drawdown,
    evaluate_bar_freshness,
    evaluate_drift,
    evaluate_guardrails,
    evaluate_reconciliation,
)
from quant.monitor.status import MonitorStatus, monitor_status_path, read_status, write_status

__all__ = [
    "GuardrailConfig",
    "GuardrailInputs",
    "GuardrailOutcome",
    "GuardrailReport",
    "MonitorStatus",
    "Severity",
    "TickResult",
    "evaluate_account_drawdown",
    "evaluate_bar_freshness",
    "evaluate_drift",
    "evaluate_guardrails",
    "evaluate_reconciliation",
    "format_heartbeat",
    "gather_inputs",
    "monitor_status_path",
    "read_status",
    "run_loop",
    "run_once",
    "write_status",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/monitor/test_exports.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant/monitor tests/monitor && uv run ruff format quant/monitor tests/monitor && uv run mypy quant/monitor`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/monitor/__init__.py tests/monitor/test_exports.py
git commit -m "feat(monitor): public API exports"
```

---

### Task 6: CLI `quant guard` (check + run)

**Files:**
- Modify: `quant/cli.py`
- Test: `tests/monitor/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/monitor/test_cli.py`:

```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

import quant.cli as cli_mod
from quant.cli import cli
from quant.governance.halt import load_halt
from quant.monitor.status import read_status


def _write_equity(data_dir: Path, values: list[float]) -> None:
    live = data_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"date": pd.bdate_range("2026-01-01", periods=len(values)), "equity": values}
    )
    df.to_parquet(live / "equity.parquet")


def test_guard_check_prints_and_never_halts(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_equity(tmp_data_dir, [90, 95, 100, 100, 100, 85, 70])  # deep drawdown
    monkeypatch.setattr(cli_mod, "_best_effort_positions", lambda settings: (None, "skipped"))
    res = CliRunner().invoke(cli, ["guard", "check"])
    assert res.exit_code == 0, res.output
    assert "account_drawdown" in res.output
    # check must NOT halt and must NOT write status
    assert load_halt(tmp_data_dir).active is False
    assert read_status(tmp_data_dir) is None


def test_guard_run_once_dry_run_writes_status_no_halt(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_equity(tmp_data_dir, [90, 95, 100, 100, 100, 85, 70])
    monkeypatch.setattr(cli_mod, "_best_effort_positions", lambda settings: (None, "skipped"))
    res = CliRunner().invoke(cli, ["guard", "run", "--once", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert load_halt(tmp_data_dir).active is False
    status = read_status(tmp_data_dir)
    assert status is not None and status.worst_severity == "halt"


def test_guard_run_once_auto_halts_on_breach(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_equity(tmp_data_dir, [90, 95, 100, 100, 100, 85, 70])
    monkeypatch.setattr(cli_mod, "_best_effort_positions", lambda settings: (None, "skipped"))
    res = CliRunner().invoke(cli, ["guard", "run", "--once"])
    assert res.exit_code == 0, res.output
    assert load_halt(tmp_data_dir).active is True
    assert "auto-halt" in load_halt(tmp_data_dir).reason
    assert "resume" in res.output.lower()
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/monitor/test_cli.py -q`
Expected: FAIL — no `guard` command / no `_best_effort_positions`.

- [ ] **Step 3: Implement — append to `quant/cli.py`**

First confirm how the TUI fetches positions: read the positions-fetch in `quant/tui.py` (`MonitorSnapshot.build`) and reuse the same `AlpacaClient` method. Then append the `guard` group. The helpers must be module-level (so tests can monkeypatch `_best_effort_positions`).

```python
@cli.group(help="Monitoring daemon (guardrails + auto kill-switch) — can HALT, never resumes.")
def guard() -> None:
    pass


def _enabled_universe_symbols() -> list[str]:
    """Union of the live-enabled strategies' universes (for bar-freshness)."""
    from quant.live.safety import enabled_strategy_slugs

    symbols: set[str] = set()
    for slug in enabled_strategy_slugs():
        symbols.update(REGISTRY[slug].spec.universe)
    return sorted(symbols)


def _best_effort_positions(settings: Settings) -> tuple[list[Any] | None, str]:  # type: ignore[type-arg]
    """Fetch Alpaca positions for reconciliation; (None, note) on any failure."""
    try:
        client = AlpacaClient(settings=settings)
        return list(client.positions()), "ok"
    except Exception as exc:  # noqa: BLE001 — recon is optional; degrade gracefully
        return None, f"alpaca unavailable: {exc!r}"


def _render_guardrail_table(report) -> Table:  # type: ignore[no-untyped-def]
    table = Table(title="Guardrails", show_header=True)
    table.add_column("Guardrail")
    table.add_column("Severity")
    table.add_column("Detail")
    palette = {"ok": "green", "warn": "yellow", "halt": "red"}
    for o in report.outcomes:
        color = palette.get(o.severity, "white")
        table.add_row(o.name, f"[{color}]{o.severity}[/{color}]", o.detail)
    return table


@guard.command("check", help="Evaluate guardrails once and print. Never halts, never writes status.")
def guard_check() -> None:
    from quant.monitor.daemon import format_heartbeat, gather_inputs
    from quant.monitor.guardrails import GuardrailConfig, evaluate_guardrails

    settings = Settings()  # type: ignore[call-arg]
    config = GuardrailConfig()
    positions, note = _best_effort_positions(settings)
    if positions is None:
        console.print(f"[yellow]reconciliation skipped — {note}[/yellow]")
    inputs = gather_inputs(
        settings.data_dir,
        asof=date.today(),
        config=config,
        alpaca_positions=positions,
        symbols=_enabled_universe_symbols(),
    )
    report = evaluate_guardrails(inputs, config)
    console.print(_render_guardrail_table(report))
    from datetime import UTC, datetime

    hb = format_heartbeat(
        inputs, report, datetime.now(UTC).replace(microsecond=0), halt_active=False
    )
    console.print(hb)
    if report.halting:
        console.print(
            "[red]A halt-severity guardrail is tripped. "
            "`quant guard run` would halt trading.[/red]"
        )


@guard.command("run", help="Run the monitoring daemon. Auto-halts on a halt verdict (unless --dry-run).")
@click.option("--interval", default=300.0, show_default=True, type=float, help="Seconds between ticks.")
@click.option("--once", is_flag=True, default=False, help="Run a single tick and exit.")
@click.option("--dry-run", is_flag=True, default=False, help="Evaluate + report but never set the halt.")
@click.option("--max-ticks", default=None, type=int, help="Stop after N ticks (default: forever).")
def guard_run(interval: float, once: bool, dry_run: bool, max_ticks: int | None) -> None:
    from quant.monitor.daemon import run_loop, run_once

    settings = Settings()  # type: ignore[call-arg]
    config = GuardrailConfig()
    symbols = _enabled_universe_symbols()

    def positions_fn() -> list[Any] | None:  # type: ignore[type-arg]
        pos, _ = _best_effort_positions(settings)
        return pos

    if once:
        res = run_once(
            settings.data_dir,
            config,
            alpaca_positions=positions_fn(),
            symbols=symbols,
            dry_run=dry_run,
        )
        console.print(res.heartbeat)
        if res.halt_triggered:
            console.print(
                "[bold red]TRADING HALTED by the monitor. "
                "Investigate, then resume with `quant governance resume --reason ...`.[/bold red]"
            )
        return

    console.print(
        f"[bold]Monitor daemon starting (interval={interval}s, dry_run={dry_run}). "
        "Ctrl-C to stop. The daemon can HALT but never resumes.[/bold]"
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
    )
```

You must also import `GuardrailConfig` inside `guard_run` (lazy-import style, matching cli.py). The constructor is `AlpacaClient(settings=settings)` and the positions method is `client.positions()` (confirmed against `quant/cli.py:699` and `quant/execution/alpaca.py:90`). `AlpacaClient` and `Any` are already imported at the top of cli.py.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/monitor/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check quant tests/monitor && uv run ruff format quant tests/monitor && uv run mypy quant`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add quant/cli.py tests/monitor/test_cli.py
git commit -m "feat(monitor): quant guard CLI (check + run) wired to the kill-switch"
```

---

### Task 7: Full-suite green + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: PASS (all prior + new monitor tests).

- [ ] **Step 2: Lint + format + types on whole repo**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy quant`
Expected: all clean. (Use `ruff format --check`, not just `ruff check`.)

- [ ] **Step 3: README note**

Read `README.md`, find the monitoring/governance section, and add:

```markdown
### Monitoring daemon (kill-switch automation)

`quant guard run` is a headless guardian loop. Each tick it evaluates
guardrails — paper-P&L drift, account drawdown, position reconciliation, bar
freshness — and **automatically pulls the kill-switch** (`set_halt`) on a
halt-severity verdict, so a bleeding or misbehaving book stops trading without
a human in the loop. It streams a one-line heartbeat and writes
`data/ops/monitor_status.json`.

Key safety property: the daemon can HALT but **never resumes** — restarting
trading is always a deliberate `quant governance resume`. Use `quant guard check`
for a one-shot, read-only evaluation (never halts), and `quant guard run --dry-run`
to observe what it *would* do without touching the kill-switch.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(monitor): document the monitoring daemon + kill-switch in the README"
```

---

## Self-Review Notes

- **Spec coverage:** guardrails (§3,§5) → Task 1; status artifact (§6) → Task 2; orchestration/run_once + heartbeat + halt asymmetry/idempotency/fail-safe (§7,§2,§9) → Task 3; run_loop fail-safe (§7,§9) → Task 4; exports → Task 5; CLI check/run (§8) → Task 6; testing strategy (§10) → Tasks 1–6.
- **Type consistency:** `GuardrailInputs` fields (`drift_rows`, `account_drawdown_pct`, `latest_equity`, `reconciliation`, `bar_freshness`) identical across guardrails.py, daemon.py, tests. `run_once(data_dir, config, *, ..., inputs=None, dry_run, now)` and `run_loop(..., inputs_fn=, alpaca_positions_fn=, sleep=, console_print=, now_fn=)` signatures consistent between daemon.py and tests. `TickResult` fields (`report, halt_triggered, halt_active, heartbeat, at`) consistent. `set_halt(data_dir, *, reason, created_at)` and `load_halt(data_dir).active` match the real `quant/governance/halt.py`.
- **No placeholders:** every step has complete code. The two verify-then-use points (the Alpaca positions-fetch method name in Task 6; `data_dir: Path` typing in Task 3) are explicit instructions to confirm against real code, not placeholders.
- **Safety invariants encoded in tests:** dry-run never halts; already-halted is idempotent and preserves the original reason; warn-only never halts; missing equity is fail-safe; loop continues on tick error; `guard check` never writes status or halts.
