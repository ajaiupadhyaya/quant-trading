# Live Reconciliation — Design

**Status:** Partial spec. Brainstorming exited after §1 at user request; remaining sections deferred.
**Date:** 2026-05-26
**Context:** Alpaca MCP server (`alpaca-paper`) added to user-level Claude Code config earlier this session. This spec is the first piece of work built on top of that integration.

## Decisions agreed in brainstorming

1. **Goal — A + B:** Use the Alpaca MCP to do (A) live-vs-backtest reconciliation, and (B) feed findings back into strategy cost/sizing models. B is a downstream spec; this doc covers the prerequisite tooling for A.
2. **Form factor — approach Z (on-demand reconciliation script):** A thin script that pulls fills from Alpaca, joins to local `trades.parquet`, and writes a dated Markdown report into `docs/live-recon/`. Not cron'd. Run on demand. Reports are committed to git so drift accumulates as a permanent record.
   - Rejected X (permanent cron'd harness): premature given live history is only 2 trading days / 57 fills.
   - Rejected Y (interactive-only): leaves no trail, findings don't accumulate.
3. **Report scope — option 4 (full picture):** slippage + timing + fidelity + per-symbol breakdown.

## §1 — Purpose & boundaries (approved)

A new on-demand script, `scripts/reconcile_live.py`, that pulls Alpaca fill data for a date range and produces a dated Markdown report comparing actual fills against backtest cost-model assumptions. The report covers slippage, timing, fidelity, and per-symbol breakdown for every order in `data/live/trades.parquet`. Reports are committed to git as a permanent audit trail of model-vs-reality drift.

**In scope:** fill outcome analysis for orders submitted via the project's own rebalance pipeline.

**Out of scope:**
- Real-time alerting (already covered by `scripts/live_watch.py`)
- Automated cron execution (Z is on-demand by design)
- Feeding findings back into strategy code (that's B, a separate later spec once enough reports accumulate)

## Project context captured during brainstorming

- Live trading history (as of 2026-05-26): **57 fills across 2 trading days** (2026-05-22, 2026-05-26). All paper. All 5 strategies enabled live since commit `4aa17da`.
- `data/live/trades.parquet` schema: `date, strategy, symbol, side, qty, client_order_id, dry_run`. Records **order intent** — does NOT record fill price, fill timestamp, commission, or partial-fill state.
- Backtest cost model (`quant/backtest/engine.py`): `slippage_bps=5.0`, `commission_bps=0.0`. Commission default is accurate for Alpaca paper; slippage is the primary thing to validate.
- `scripts/live_watch.py` is a real-time event monitor (equity/position/PnL deltas → stdout for Claude Monitor tool). Does NOT do per-fill reconciliation. Complementary to this work, not overlapping.
- `client_order_id` format from rebalance code: `{strategy}-{YYYYMMDD}-{SYMBOL}-{8hex}` — gives a clean join key between local `trades.parquet` and Alpaca's order history.

## Deferred / open design questions

These were not resolved before the user requested exit. Anyone picking this up should answer them before implementation:

1. **Signal-price source for slippage comparison.** `trades.parquet` doesn't record the bar-close price that drove the signal. Options:
   - (a) Re-fetch bar close from the data layer at recon time using the trade date
   - (b) Extend `trades.parquet` to record signal price at submission time (requires touching rebalance code)
   - (c) Use the bar close on the trade date as an approximation
2. **State management.** How does the script know "since when"? Options: high-watermark file, always-rewrite idempotent reports, explicit `--since` CLI flag.
3. **Report file naming.** `docs/live-recon/YYYY-MM-DD.md` (per-run-day) vs. `docs/live-recon/rebalance-{run_id}.md` (per-rebalance) vs. weekly rollups.
4. **Commit policy for reports.** Auto-commit on script run, or leave staged for user review?
5. **MCP vs. direct SDK.** The project already has `quant/execution/alpaca.py` (an `AlpacaClient` wrapper). Should `reconcile_live.py` use that, or use the newly-added MCP server? (MCP is for interactive sessions; the script should probably use the existing SDK wrapper for headless runs.)
6. **Tear-sheet integration.** Should reconciliation findings ever surface in the existing tear-sheet pipeline, or stay separate in `docs/live-recon/`?

## Next steps when resumed

Re-enter brainstorming, work through §§2–6 (architecture, data flow, error handling, testing, file layout), resolve the deferred questions above, then `superpowers:writing-plans` for implementation.
