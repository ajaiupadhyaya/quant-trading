# Scenario / Stress-Shock Evaluation (Raise-the-Ceiling Phase 2) — 2026-06-05

## Goal

Add an institutional **scenario / stress-shock** view to the live book: "how much
would today's holdings lose under a 2008-style crash, a +100bp rate shock, etc.?"

This is the next unbuilt item in `docs/specs/2026-06-02-raise-the-ceiling-roadmap.md`
Phase 2 ("Scenario/stress-shock evaluation (`quant/risk/scenarios.py`) as a
separate WARN metric"). The portfolio risk gate (VaR/CVaR/vol/beta/asset-class)
and its Guard-5 wiring already shipped; this adds the stress dimension.

## Non-negotiable constraints (inherited from the roadmap)

- **Read-only / WARN-only.** No order-path enforcement. Never mutates `netted`,
  never halts, never de-authorizes a strategy. The Guard-5 BLOCK branch is
  human-gated and out of scope here.
- **Fail-open.** Every live entry point (Guard 5, brief, CLI) tolerates any
  failure by logging and continuing. A bug in stress code can never abort a
  rebalance or page.
- **Pure core.** The shock math is a pure function of `(weights, returns/shocks)`
  so it is unit-testable with no data/network — mirrors `compute_portfolio_risk`.
- **Defensive-etf records OK every run.** Default limits leave the live sleeve
  passing; a recompute must never flip it to a violation by construction.

## Architecture

### 1. `quant/risk/scenarios.py` (new pure module)

Mirrors `quant/risk/portfolio.py`'s shape: frozen dataclasses, a pure compute
function, a best-effort `live_*` wrapper, and `render()`.

Both scenario kinds reduce to the same kernel — **apply a per-asset return shock
to current signed weights → portfolio P&L%**:

```
pnl_pct = Σ_i  weight_i * shock_i
```

- `HistoricalScenario(name, start, end, description)` — `shock_i` = asset *i*'s own
  cumulative simple return over `[start, end]`, computed from its price history.
  Assets with no coverage in the window are excluded and reported in
  `missing_symbols` (graceful degrade, never raise).
- `HypotheticalScenario(name, shocks, description)` — `shocks` maps an
  asset-class bucket (via the existing `_SECTOR_MAP`: equity/bond/gold/commodity/
  real_estate/other) **or** a specific symbol → a return shock. **Symbol-level keys
  override class-level keys** so a rate scenario can respect duration
  (`TLT −15%, IEF −7%`). A held asset whose class/symbol is absent from `shocks`
  contributes 0 (no shock) and is noted.

Dataclasses (all frozen):

- `ScenarioResult(name, kind, pnl_pct, by_class: dict[str,float], missing_symbols:
  tuple[str,...], computable: bool)` — `by_class` is the per-asset-class P&L
  contribution (for explainability). `pnl_pct` negative = loss.
- `StressReport(results: tuple[ScenarioResult,...], worst_loss: float | None,
  worst_scenario: str | None, computable: bool, degraded: tuple[str,...])` —
  `worst_loss` is the most negative `pnl_pct` expressed as a positive loss
  fraction (parallels `var_95`'s sign convention: positive = loss); `None` if
  nothing computable.

Functions:

- `default_scenarios() -> tuple[Scenario, ...]` — the curated library (below).
- `compute_stress(weights, returns, scenarios) -> StressReport` — **pure.**
  `returns` is a daily-returns panel (rows=dates, cols=symbols) used by historical
  scenarios; hypothetical scenarios ignore it. Degenerate inputs degrade per
  scenario rather than raising.
- `live_stress(positions, equity, *, asof, lookback_days=180) -> StressReport |
  None` — best-effort: reuse the same `get_bars` + `weights_from_positions` path
  as `live_portfolio_risk` (`lookback_days` matches its 180 default and is used
  for current weights); historical scenarios fetch their own (older) windows.
  Returns `None` on flat book / any data failure.
- `StressReport.render() -> str` — compact one-block summary (worst scenario +
  loss, then per-scenario P&L) for CLI/brief/Slack.

**Curated default library:**

Historical replays (windows are peak→trough of the episode):
- `2008-GFC` — 2008-09-01 → 2009-03-09 (Lehman → bottom)
- `2020-COVID` — 2020-02-19 → 2020-03-23
- `2022-rate-selloff` — 2022-01-01 → 2022-10-14 (60/40's worst year)
- `2013-taper-tantrum` — 2013-05-22 → 2013-06-24

Hypothetical shock vectors (class-level unless symbol noted):
- `equity-crash-20` — equity −20%, real_estate −25%, commodity −10%, gold +5%, bond +5%
- `rate-shock-+100bp` — TLT −15%, IEF −7% (symbol overrides), equity −5%, real_estate −10%, gold −3%
- `stagflation` — commodity +15%, gold +10%, bond −10%, equity −10%
- `risk-off-flight` — equity −15%, gold +8%, bond +5%, commodity −10%, real_estate −12%

### 2. Gate integration (Guard 5, `quant/live/rebalance.py`) — WARN-only

Inside the existing fail-open Guard-5 `try/except`, after the current
`build_portfolio_risk_gate(...)`:

1. `stress = live_stress(post_trade, account.equity, asof=asof)` (degraded
   placeholder if `None`, like the existing `port_risk` fallback).
2. Fold the `StressReport` into the **same** `portfolio_risk_gate.<date>.json`
   artifact under a new `"stress"` key (`_write_portfolio_risk_gate_artifact`
   gains an optional `stress` arg; absent → key omitted, back-compatible).
3. If `stress.worst_loss is not None and worst_loss > limits.max_scenario_loss`,
   append a `RiskViolation("stress", ...)` to the gate result before recording the
   `CheckResult`. In WARN mode this only records; the existing BLOCK branch (human-
   gated, default OFF) already handles any violation uniformly — no new block path.

`netted` is read-only throughout; the whole block stays inside the existing
fail-open guard.

### 3. Limit (`PortfolioRiskLimits`)

Add `max_scenario_loss: float = 0.30`. Calibrated so the defensive sleeve's worst
historical/hypothetical loss stays under it with headroom (a 100%-defensive book
is shallow-drawdown by design). Validated against the live book during
implementation; widen the default if the real worst-case is close.

### 4. CLI `quant risk scenarios`

New Click command mirroring `risk_portfolio`: fetch the live book from Alpaca, run
`live_stress`, render a table (scenario → P&L%, worst highlighted), and write
`data/risk/scenarios.<date>.json`. Flat book / no history → friendly message,
exit 0.

### 5. Analyst brief (`quant/analyst/context.py`)

Add a `stress: Any | None` field to `AnalystContext`, a `_stress(positions,
equity, asof)` helper (best-effort, mirrors `_portfolio_risk`), populate it in the
context builder, and append one render line (`"Stress (worst): "+report.render()`)
guarded by `contextlib.suppress(Exception)` — exactly the `portfolio_risk` pattern.

## Data flow

```
positions (broker book + netted deltas)  ─┐
equity ───────────────────────────────────┤→ live_stress() → StressReport
bars (held assets, incl. historical windows)┘        │
                                                      ├→ gate artifact "stress" key
                                                      ├→ CheckResult "portfolio_risk_gate"
                                                      ├→ quant risk scenarios (CLI table + json)
                                                      └→ analyst brief render line
```

## Error handling

- Pure core: degenerate weights/returns → per-scenario `computable=False` +
  `missing_symbols`; never raises.
- `live_stress`: any exception → `logger.info` + `None` (analysis convenience).
- Guard 5: already wrapped in fail-open `try/except`; stress code adds nothing
  that can escape it.
- CLI: Alpaca/data failures → `click.ClickException` or friendly message; no stack
  trace to the user.

## Testing (TDD — tests first)

Pure unit tests (`tests/risk/test_scenarios.py`), synthetic weights/returns:
- Historical replay: cumulative-return math over a window; missing-symbol exclusion
  populates `missing_symbols` and still computes over the rest.
- Hypothetical: class-level resolution; **symbol override beats class**; an
  unshocked held asset contributes 0.
- `worst_loss`/`worst_scenario` selection (most-negative pnl) and sign convention
  (positive = loss).
- Limit → violation: `worst_loss > max_scenario_loss` yields a `stress`
  `RiskViolation`; at/under the limit yields none; `None` worst never violates.
- `render()` is non-empty and degrades cleanly with `None` fields.

Integration / regression:
- Guard 5 with a defensive-etf-like book records a `stress` CheckResult `ok` and
  leaves `netted` byte-identical in WARN mode (extend the existing rebalance test).
- Artifact JSON gains a well-formed `stress` key; absent stress → key omitted
  (back-compat).
- Full suite green (`uv run pytest`).

## Out of scope (explicit)

- WARN→BLOCK flip for stress (human-gated, roadmap Phase 4).
- Factor-model propagation (equity beta / rate duration loadings) — the
  symbol-override mechanism covers the duration nuance without a factor model.
- Any change to live `spec.universe`, `enable_live`, or governance.

## Operational note

This repo's working tree is the live launchd host's tree (M4). Implement on a
separate git worktree/branch, keep every commit's suite green, and merge to `main`
deliberately. The change is read-only + fail-open, so it cannot affect order flow,
but the branch discipline keeps the live tree stable during development.
