# Quant Platform Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the project from evidence-gated paper trading to a durable autonomous quant research and operations platform that can safely improve strategies without weakening governance.

**Architecture:** Keep the current local-first CLI/TUI architecture. Add reproducibility metadata, complete validation evidence, execution-cost telemetry, governance v2 allocation/drift controls, and a controlled research sandbox inspired by Awesome Quant resources. No new strategy or ML/RL model may affect live paper capital until it passes the existing governance gate.

**Tech Stack:** Python 3.12, Click/Rich CLI, pandas/parquet, existing walk-forward/CPCV/bootstrap validation, Alpaca paper execution, GitHub Actions, optional research-only evaluations of libraries listed in Awesome Quant.

---

## Source Inputs

- Current repo state: `main` at `13869d6` plus governance fail-closed hardening at `8de9cd2`.
- Governance state: all strategies are currently quarantined, mostly because bootstrap lower-5% gates or validation evidence are missing.
- Deferred follow-ups: `docs/superpowers/specs/2026-05-26-deferred-followups.md`.
- Awesome Quant reference map: https://github.com/wilsonfreitas/awesome-quant

Awesome Quant resources to evaluate, not blindly adopt:

- `backtester-mcp`: useful as a reference for overfitting checks such as PBO, DSR, bootstrap CI, and walk-forward.
- `skfolio`, `PyPortfolioOpt`, `Riskfolio-Lib`: candidates for evidence-weighted allocation and CVaR/portfolio stress checks.
- `alphalens` / `alphalens-reloaded`: candidates for factor diagnostics on multi-factor and momentum signals.
- `Qlib`: benchmark for an end-to-end ML quant pipeline; use as architecture reference before any dependency decision.
- `FinRL`: RL sandbox reference only; no live pathway until governance has enough paper-P&L history.
- `NautilusTrader`, `Lean`, `StrateQueue`: execution architecture references; do not replace the existing engine unless a measured gap justifies it.

---

## Task 1: Validation Reproducibility And Bootstrap Regression Audit

**Files:**
- Create: `quant/governance/audit.py`
- Create: `tests/governance/test_audit.py`
- Modify: `quant/cli.py`
- Modify: `README.md`

- [ ] Add a `ValidationAudit` dataclass recording strategy slug, git SHA, validation command, data range, bootstrap seed, bootstrap resamples, chosen params hash, walkforward parquet hash, and validation report hash.
- [ ] Add `quant governance audit <strategy>` to print reproducibility metadata and explain why a strategy is quarantined.
- [ ] Add tests for deterministic hash calculation and missing artifact reporting.
- [ ] Run `quant validate trend --bootstrap-resamples 5000` and `quant validate momentum --bootstrap-resamples 5000` after confirming runtime is acceptable.
- [ ] Document whether the negative bootstrap lower-5% results are stable or sampling noise.

**Acceptance:** Trend and momentum have an auditable explanation for the bootstrap gate regression. Governance remains fail-closed if the regression persists.

---

## Task 2: Complete Missing Strategy Evidence

**Files:**
- Modify only generated artifacts under `data/backtests/<slug>/` and `data/governance/`
- Modify: `docs/notes/2026-05-25-go-live-decisions.md`

- [ ] Re-run `quant validate multi-factor`, using the new EDGAR memoization.
- [ ] Re-run `quant validate pairs` with conservative runtime settings first, then full settings if feasible.
- [ ] Re-run `quant validate risk-parity`.
- [ ] Run `quant governance refresh`.
- [ ] Update the go-live decisions note with final evidence for all five strategies.

**Acceptance:** No live-capable strategy is missing validation evidence. Strategies may remain quarantined; missing evidence should not be the reason.

---

## Task 3: Execution-Cost Telemetry

**Files:**
- Create: `quant/live/execution_cost.py`
- Create: `tests/live/test_execution_cost.py`
- Modify: `quant/live/recon.py`
- Modify: `quant/live/recon_render.py`
- Modify: `scripts/reconcile_live.py`

- [ ] Add fill-time mid-price lookup from Alpaca minute bars where available.
- [ ] Add `execution_cost_bps` separate from existing signal-to-fill drift.
- [ ] Keep missing intraday data as an explicit `no_mid_price` status, not a silent zero.
- [ ] Extend reconciliation reports with per-strategy execution-cost summaries.
- [ ] Add tests for buy/sell cost sign handling, missing mid data, and aggregate summaries.

**Acceptance:** Live reconciliation distinguishes signal drift from execution cost and can support future per-strategy slippage calibration.

---

## Task 4: Governance V2 Capital Allocation

**Files:**
- Create: `quant/governance/allocation.py`
- Create: `tests/governance/test_allocation.py`
- Modify: `quant/live/rebalance.py`
- Modify: `quant/cli.py`

- [ ] Add a capital allocation model with three modes: equal-live, DSR-weighted, and capped evidence score.
- [ ] Keep equal-live as the default until at least two strategies are `live`.
- [ ] Add max allocation cap per strategy, default 40%.
- [ ] Add min allocation floor for live strategies, default 5%, only when total capital permits.
- [ ] Render allocation weights in `quant governance status`.
- [ ] Use allocation weights in live rebalance instead of equal split once mode is enabled.

**Acceptance:** Allocation is deterministic, tested, and cannot allocate to quarantined strategies.

---

## Task 5: Paper-P&L Drift Monitor

**Files:**
- Create: `quant/governance/drift.py`
- Create: `tests/governance/test_drift.py`
- Modify: `quant/cli.py`
- Modify: `quant/tui.py`

- [ ] Compare realized strategy P&L from `data/live/` against backtest expectation over rolling 5/20/60 trading-day windows.
- [ ] Add z-score style drift flags: `normal`, `watch`, `halt_candidate`.
- [ ] Add `quant governance drift` CLI output.
- [ ] Show drift status in TUI strategy rows.
- [ ] Do not auto-halt in this task; emit evidence for human review first.

**Acceptance:** A strategy can be flagged when paper behavior diverges from validation without changing live eligibility automatically.

---

## Task 6: Awesome Quant Research Intake

**Files:**
- Create: `docs/research/awesome-quant-evaluation.md`
- Create: `scripts/evaluate_quant_resource.py`
- Create: `tests/test_resource_evaluation.py`

- [ ] Build a rubric: relevance, maintenance, license, dependency weight, overlap with existing code, testability, and failure blast radius.
- [ ] Evaluate `skfolio`, `Riskfolio-Lib`, `alphalens-reloaded`, `Qlib`, `FinRL`, `vectorbt`, and `backtester-mcp`.
- [ ] For each, choose one of: `adopt now`, `research sandbox`, `reference only`, `reject for now`.
- [ ] Add no runtime dependency unless it has a concrete task and measurable acceptance criteria.

**Acceptance:** Awesome Quant becomes a curated project-specific intake document, not a dependency dump.

---

## Task 7: Weekly Automation

**Files:**
- Create: `.github/workflows/weekly-validation-governance.yml`
- Modify: `README.md`

- [ ] Add a weekly workflow that validates one strategy matrix at a time with timeout limits.
- [ ] Commit updated `validation_report.json`, `chosen_params.json`, walkforward parquet, and governance artifacts.
- [ ] Add workflow dispatch inputs for strategy slug and bootstrap resamples.
- [ ] Keep daily rebalance fail-closed if weekly validation has not produced fresh passing evidence.

**Acceptance:** Governance freshness no longer depends on operator memory.

---

## Task 8: Final Verification And Operator Runbook

**Files:**
- Create: `docs/runbooks/autonomous-paper-trading.md`
- Modify: `README.md`

- [ ] Document startup, daily checks, weekly validation, quarantine interpretation, emergency stop, and recovery.
- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run mypy quant`.
- [ ] Run `uv run quant governance status`.
- [ ] Run `uv run quant rebalance --dry-run`.
- [ ] Push final branch to GitHub.

**Acceptance:** A future session can operate the system from the runbook without relying on chat history.

---

## Execution Order

1. Task 1: explain current quarantines before trying to unlock anything.
2. Task 2: complete evidence across all strategies.
3. Task 3: improve live cost observability.
4. Task 4 and Task 5: add allocation and drift monitoring.
5. Task 6: evaluate Awesome Quant resources against actual gaps.
6. Task 7 and Task 8: automate and document operations.

## Non-Negotiables

- No quarantined strategy receives live paper capital.
- No ML/RL model reaches live paper capital in this plan.
- No external framework replaces the local engine without benchmark evidence.
- Real-money deployment remains out of scope.
