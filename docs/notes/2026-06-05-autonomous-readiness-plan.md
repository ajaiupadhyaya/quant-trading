# Autonomous Readiness Plan And 2026-06-05 Verification

## Goal

Make `quant-trading` a 24/7 paper-trading quantitative research and analyst
system on the M4 Mac mini: deterministic quant engines make routine decisions,
governance and risk gates remain fail-closed, and Claude/LLM calls stay
read-only or explicitly human-gated unless a later promotion gate approves more.

## Current State After Sync

Local `main` was fast-forwarded to `origin/main` at `6b73a69`, which already
contains the merged M4/continuous-analyst work:

- M4 launchd deployment (`quant/deploy`, `deploy/`) with tick, guard, and engine
  plists.
- Continuous read-only engine (`quant/engine`) and analyst/watch layer
  (`quant/analyst`).
- Broader data/model surface: macro nowcast, news/sentiment, event risk,
  fundamentals, options vol surface, portfolio risk, signals engine, and
  forecast models.
- Phase 8 honesty remains intact: HAR-RV volatility is advisory; macro-regime,
  cross-sectional factor, and stacking ensemble results remain research-only
  where they failed robust validation.

## Plan

1. Keep the M4-only deployment as the operational spine: launchd tick, guard,
   and read-only engine supervised locally; GitHub Actions stay manual fallback.
2. Treat `quant doctor`, live-universe data quality, and pretrade risk as the
   minimum daily readiness gate.
3. Preserve the deterministic/LLM split: Tier 0 quant and Tier 1 governance make
   routine decisions; Claude summarizes, explains, and proposes.
4. Promote new models or actuators only through the existing evidence gates:
   walk-forward, CPCV, DSR/PSR, bootstrap, regime stress, costs, and drift.
5. Next research frontier after readiness is stable: capacity reporting, then
   only validated volatility/portfolio/execution improvements; no overfit model
   gets paper capital by lowering thresholds.

## Fixes Applied In This Pass

- `last_strategy_positions` now uses explicit `snapshot_id` values for new
  strategy-position snapshots and falls back to latest date for legacy parquet
  files. This prevents stale prior-session symbols from reappearing in
  reconciliation; it fixed the false GLD mismatch observed on 2026-06-05.
- `quant data quality` now exits with code `2` when the report writes
  `"passed": false`, so scheduler/CI callers cannot silently accept bad bars.
- Forecast modules were made strict-mypy clean without changing model math:
  numpy outputs are explicitly typed and pandas correlations handle `None`.
- Recent live ETF bars were refreshed through 2026-06-04.

## Verification Snapshot

- `uv run ruff check .` -> passed.
- `uv run mypy quant` -> passed.
- `uv run pytest` -> `1087 passed, 1 warning`.
- `uv run quant data quality --start 2018-01-01 --symbols SPY,TLT,IEF,GLD,DBC,VNQ,EFA,EEM` -> passed; zero missing bars, zero duplicate timestamps, zero bad OHLC, not stale.
- `uv run quant doctor` -> `7/7 checks passed`; Alpaca paper connectivity OK, governance live strategy is `defensive-etf-allocation`, bar freshness current, reconciliation all 3 broker symbols within tolerance, risk limits within budget.
- `uv run quant risk pretrade` -> passed; wrote `data/risk/pretrade_report.json` and `data/risk/portfolio_risk_gate.2026-06-05.json`.

## Remaining Watch Items

- Keep the pre-sync stash `stash@{0}` (`codex-pre-autonomous-sync-2026-06-05`)
  until any local-only artifacts in it are reviewed or intentionally dropped.
- The account is paper-only. Real-money deployment remains a separate explicit
  decision and should require a new gate, not an assumption.
- Data provider limitations remain visible: Alpaca recent SIP restrictions can
  force yfinance fallback; Wikipedia S&P 500 fetch can 403. Existing commands
  handled both during this pass, but these are operational dependencies to watch.
