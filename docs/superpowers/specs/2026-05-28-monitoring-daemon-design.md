# Autonomous Monitoring Daemon + Kill-Switch Design (Pillar 2)

**Date:** 2026-05-28
**Status:** Approved (autonomous build — user waived approval gates for pillars 2–4)
**Pillar:** 2 of 4 (regime detector → **monitoring daemon** → options/Greeks → position sizing)

## 1. Goal

A headless **guardian loop** that, on a fixed interval, evaluates a set of
**guardrails** against the live paper book and **automatically pulls the
existing kill-switch** (`quant/governance/halt.py`) when a halt-severity
guardrail trips. It streams a one-line **heartbeat** + alerts to the terminal
each tick and writes a machine-readable **status artifact**.

This is purely an **orchestration + automation** layer. Every building block
already exists — the kill-switch, drift detection (with `watch_z`/`halt_z`
thresholds), account-drawdown risk checks, reconciliation, bar-freshness, the
TUI. What is missing, and what this pillar adds, is the thing that **ties them
together and acts**: today `summarize_drift` produces `halt_candidate` flags
that nothing consumes, and `check_risk_limits` returns a `halted_strategies`
set that nothing applies. The daemon closes that gap.

## 2. The core safety asymmetry

**The daemon can HALT but never RESUME.** Auto-halting is safe (it stops
trading; the existing `run_rebalance` already fail-closes on an active halt).
Auto-resuming is not — restarting trading must always be a deliberate human act
via `quant governance resume`. This asymmetry is the central design invariant
and is enforced structurally: the daemon only ever calls `set_halt`, never
`clear_halt`.

A second invariant: the daemon's **only write actions** are `set_halt` (the
kill-switch JSON) and the status artifact. It never places, cancels, or
modifies orders. It is a watcher with one lever.

## 3. Guardrails

A **guardrail** evaluates one aspect of book health and returns a
`GuardrailOutcome(name, severity, detail)` where `severity ∈ {ok, warn, halt}`.

| Guardrail | Source | Severity rule | Halt authority? |
|-----------|--------|---------------|-----------------|
| `drift` | `summarize_drift` on account equity returns | any row `halt_candidate` → halt; any `watch` → warn; else ok | **yes** |
| `account_drawdown` | `_recent_drawdown_pct` on `equity.parquet` | `dd ≤ -budget.max_drawdown` → halt; else ok | **yes** |
| `reconciliation` | `check_reconciliation` (needs Alpaca positions) | breach → warn (or halt if `reconciliation_is_halt`); no account → ok ("skipped") | no (default) |
| `bar_freshness` | `check_bar_freshness` | stale → warn; else ok; no data → warn | no |

**Why drift + drawdown are the halt triggers and reconciliation/freshness are
warn-only:** drift and drawdown are computed from the local, authoritative
`equity.parquet` (appended every rebalance) — they are stable, reproducible, and
genuinely indicate the book is bleeding or behaving abnormally. Reconciliation
mismatches and stale bars often have benign, transient causes (mid-day partial
fills, a long-weekend cache gap) and auto-halting on them would be jumpy. They
are surfaced loudly as warnings; `reconciliation_is_halt=True` is available for
operators who want the stricter posture. (`max_drawdown`/`halt_z` thresholds are
deliberately reused from the existing `StrategyRiskBudget` / `DriftConfig` so
there is one canonical definition of "too far.")

The overall tick **halts** iff any guardrail returns `halt`.

## 4. Package layout

New package `quant/monitor/`:

| File | Responsibility |
|------|----------------|
| `guardrails.py` | Pure: `Severity`, `GuardrailOutcome`, `GuardrailConfig`, `GuardrailInputs`, `GuardrailReport`, the four `evaluate_*` functions + `evaluate_guardrails`. No I/O. |
| `status.py` | `MonitorStatus` dataclass + `monitor_status_path`, `write_status`, `read_status`. |
| `daemon.py` | I/O + orchestration: `gather_inputs`, `run_once`, `run_loop`, `TickResult`, `format_heartbeat`. |
| `__init__.py` | Curated exports. |

CLI: a new `quant guard` group with `check` and `run`.

Tests under `tests/monitor/`. No new dependencies (reuse rich, pandas, and the
existing governance/live modules).

## 5. Pure guardrail layer (`quant/monitor/guardrails.py`)

```python
Severity = Literal["ok", "warn", "halt"]
_RANK = {"ok": 0, "warn": 1, "halt": 2}

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
    account_drawdown_pct: float            # non-positive
    reconciliation: CheckResult | None     # None => skipped (no live account)
    bar_freshness: CheckResult | None      # None => skipped

@dataclass(frozen=True)
class GuardrailReport:
    outcomes: list[GuardrailOutcome]
    @property
    def worst_severity(self) -> Severity: ...   # max by _RANK, "ok" if empty
    @property
    def halting(self) -> bool: ...              # worst_severity == "halt"
```

Evaluators (all pure, total, never raise):

- `evaluate_drift(rows, *, name="drift") -> GuardrailOutcome` — `halt` if any
  `row.flag == "halt_candidate"` (detail lists offending strategy/windows),
  `warn` if any `watch`, else `ok`. Empty rows → `ok` ("no drift history").
- `evaluate_account_drawdown(dd_pct, budget) -> GuardrailOutcome` — `halt` if
  `dd_pct <= -abs(budget.max_drawdown)`, else `ok`. Detail formats the dd vs cap.
- `evaluate_reconciliation(recon, *, halt_on_breach) -> GuardrailOutcome` —
  `None` → `ok` ("skipped: no account"); `recon.ok` → `ok`; else
  `halt if halt_on_breach else warn` with `recon.detail`.
- `evaluate_bar_freshness(freshness) -> GuardrailOutcome` — `None` → `ok`
  ("skipped"); `freshness.ok` → `ok`; else `warn` with `freshness.detail`.
- `evaluate_guardrails(inputs, config) -> GuardrailReport` — runs all four in a
  fixed order and returns the report.

## 6. Status artifact (`quant/monitor/status.py`)

Path: `{data_dir}/ops/monitor_status.json`.

```python
@dataclass(frozen=True)
class MonitorStatus:
    version: int          # 1
    at: datetime
    worst_severity: Severity
    halt_triggered_this_tick: bool
    halt_active: bool
    outcomes: list[GuardrailOutcome]
    heartbeat: str
```

`write_status(data_dir, status)` writes pretty JSON (sorted keys, trailing
newline, `outcomes` as a list of `{name, severity, detail}`); `read_status`
parses it back (returns `None` if absent). This is the daemon's
machine-readable channel — the TUI or external tooling can read it to show
"guardian: last tick HH:MM, all ok / HALTED".

## 7. Orchestration (`quant/monitor/daemon.py`)

```python
@dataclass(frozen=True)
class TickResult:
    report: GuardrailReport
    halt_triggered: bool     # set_halt was called THIS tick
    halt_active: bool        # halt is active after this tick
    heartbeat: str
    at: datetime

def gather_inputs(
    data_dir: Path, *, asof: date,
    alpaca_positions: list[PositionRow] | None = None,
    symbols: list[str] | None = None,
) -> GuardrailInputs:
    """I/O side. Reads equity.parquet -> drift rows + drawdown; builds recon
    (only if alpaca_positions given) and bar-freshness CheckResults."""

def run_once(
    data_dir: Path, config: GuardrailConfig, *,
    asof: date | None = None,
    inputs: GuardrailInputs | None = None,   # injectable for tests
    alpaca_positions: list[PositionRow] | None = None,
    symbols: list[str] | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> TickResult:
    """One tick. If inputs is None, gather_inputs(...) is called. Evaluate
    guardrails; if report.halting and not already halted and not dry_run,
    set_halt(reason="auto-halt: <names>"). Always write status. Return TickResult."""

def run_loop(
    data_dir: Path, config: GuardrailConfig, *,
    interval_s: float = 300.0, dry_run: bool = False,
    max_ticks: int | None = None,
    alpaca_positions_fn: Callable[[], list[PositionRow]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    console_print: Callable[[str], None] | None = None,
) -> list[TickResult]:
    """Thin: repeatedly run_once + print heartbeat, sleeping interval_s between
    ticks. Stops after max_ticks (None = forever). `sleep`/`console_print`/
    `alpaca_positions_fn` are injectable so the loop is testable without real
    time or a live account."""
```

**Halt idempotency:** if `load_halt(data_dir).active` is already `True`,
`run_once` does NOT call `set_halt` again (`halt_triggered=False`) but the
status still reports `halt_active=True`. A halting verdict while already halted
is logged but does not rewrite the artifact (preserves the original halt reason
and timestamp).

**Heartbeat** (`format_heartbeat`): one line, e.g.
`14:32:05 | equity $1,003,210 dd -1.2% | drift ok | recon ok | bars ok` and when
halting `… | ⛔ HALT(auto-halt: account_drawdown)`. Built from equity tail +
outcomes; pure function of `(inputs, report, halt_state, now)`.

## 8. CLI: `quant guard`

```
quant guard check        # evaluate once, print table + heartbeat. NEVER halts,
                         # NEVER writes status. Safe to run anytime (inspection).

quant guard run [--interval 300] [--once] [--dry-run] [--max-ticks N]
                         # the daemon. Each tick: evaluate, print heartbeat,
                         # write status, and auto-set the kill-switch on a
                         # halt verdict UNLESS --dry-run. --once = single tick.
```

- `guard check` constructs `GuardrailConfig()`, calls `run_once(..., dry_run=True)`
  but prints the guardrail table; to honor "never writes status" it uses a
  report-only path (`gather_inputs` + `evaluate_guardrails`) rather than the
  status-writing `run_once`. It best-effort fetches Alpaca positions for the
  reconciliation row; on any Alpaca error it prints a note and proceeds with
  recon skipped.
- `guard run` builds an `alpaca_positions_fn` closure (best-effort; returns
  `[]`/skips on error) and calls `run_loop` (or a single `run_once` if `--once`).
  On a real (non-dry-run) auto-halt it prints a prominent red alert telling the
  operator that trading is halted and how to resume.

No registry logging here (this is operational monitoring, not an experiment).

## 9. Point-in-time / safety considerations

- The two halt triggers read only `equity.parquet`, which is appended *after*
  each rebalance from Alpaca's reported account equity — there is no look-ahead
  concern (it is realized history).
- The daemon is **fail-safe, not fail-deadly**: if `gather_inputs` cannot read
  equity (empty/missing), drift and drawdown evaluate to `ok` (no false halt),
  and bar-freshness/recon degrade to skip/warn. A monitoring failure must not
  itself trigger a halt, and conversely must not crash the loop — `run_loop`
  catches per-tick exceptions, emits a `warn` heartbeat, and continues.
- Because only `trend` is currently `enabled_live` and rebalance is already
  fail-closed, enabling this daemon is strictly additive safety.

## 10. Testing strategy

- **guardrails.py:** each `evaluate_*` across ok/warn/halt/skip; `evaluate_guardrails`
  worst-severity aggregation and `halting`; empty-input neutrality.
- **status.py:** write→read round-trip incl. outcomes and datetime.
- **daemon.py (the heart):** drive `run_once` with synthetic `equity.parquet` (or
  injected `inputs`):
  - clean book → no halt, status written, `halt_triggered=False`.
  - drawdown breach → `set_halt` called, `halt.json` active, `halt_triggered=True`.
  - drift `halt_candidate` → halt.
  - already halted → `halt_triggered=False`, original halt reason preserved.
  - `dry_run=True` with a halting verdict → `set_halt` NOT called (`halt.json`
    inactive), status still records `worst_severity="halt"`.
  - warn-only (stale bars / recon breach, `reconciliation_is_halt=False`) → no halt.
  - `gather_inputs` with missing equity → all-ok inputs (fail-safe).
  - `run_loop` with `max_ticks=2`, injected `sleep`/`console_print` → two ticks,
    no real sleep; a tick that raises is caught and the loop continues.
- **CLI:** `guard check` smoke (prints table, writes NO `halt.json`, NO status);
  `guard run --once --dry-run` (writes status, `halt.json` stays inactive);
  `guard run --once` with a breach equity fixture (sets `halt.json` active).
  Uses `tmp_data_dir` + `fake_env`; Alpaca access monkeypatched/skipped.

## 11. Out of scope (explicit)

- **External alerting** (Slack/email/webhooks/push). The terminal heartbeat +
  status JSON are the channels; richer alerting is a follow-on and must not add
  dependencies to core.
- **Auto-resume** — never. Resume is always manual.
- **Per-strategy equity attribution** for drawdown (inherits the existing
  account-level conservatism from `check_risk_limits`).
- **Improving the drift "expected returns" model** (currently `expected=0`,
  matching `quant governance drift`) — a modeling refinement, tracked separately.
- **A daemon status pane in the TUI** — cheap follow-on; the status artifact is
  designed to make it trivial later.
- **Process supervision** (systemd/launchd/pm2). The daemon is a foreground loop;
  wrapping it in a supervisor is an ops concern, not code.

## 12. Relationship to existing code

- Kill-switch: `quant/governance/halt.py` (`set_halt`, `load_halt`) — the daemon's
  only lever. Honored already by `quant/live/rebalance.py`.
- Drift: `quant/governance/drift.py` (`summarize_drift`, `DriftConfig`, `DriftRow`).
- Risk/recon/freshness: `quant/live/safety.py` (`check_reconciliation`,
  `_recent_drawdown_pct`, `StrategyRiskBudget`, `check_bar_freshness`, `CheckResult`).
- Equity: `quant/live/bookkeeping.py` (`read_equity`).
- The TUI (`quant/tui.py`) remains the human dashboard; this daemon is the
  headless guardian. They share artifacts (halt.json, equity.parquet) and the
  new `ops/monitor_status.json` bridges them.
