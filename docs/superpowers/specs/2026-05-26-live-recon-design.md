# Live Reconciliation — Design

**Status:** Approved 2026-05-26.
**Context:** Alpaca MCP server (`alpaca-paper`) was added to user-level Claude Code config earlier this session. This spec is the first piece of work built on top of that integration — though, per §2, the headless reconciliation script uses the existing SDK wrapper, not the MCP. The MCP remains available for interactive investigation sessions.

## Decisions agreed in brainstorming

1. **Goal — A + B:** Use Alpaca to do (A) live-vs-backtest reconciliation, and (B) feed findings back into strategy cost/sizing models. B is a downstream spec; this doc covers the prerequisite tooling for A.
2. **Form factor — approach Z (on-demand reconciliation script):** A thin script that pulls fills from Alpaca, joins to local `trades.parquet`, and writes a dated Markdown report into `docs/live-recon/`. Not cron'd. Run on demand. Reports are committed to git (by hand, not auto) so drift accumulates as a permanent record.
   - Rejected X (permanent cron'd harness): premature given live history is only 2 trading days / 57 fills.
   - Rejected Y (interactive-only): leaves no trail, findings don't accumulate.
3. **Report scope — option 4 (full picture):** slippage + timing + fidelity + per-symbol breakdown.

## Project context captured during brainstorming

- Live trading history (as of 2026-05-26): **57 fills across 2 trading days** (2026-05-22, 2026-05-26). All paper. All 5 strategies enabled live since commit `4aa17da`.
- `data/live/trades.parquet` schema: `date, strategy, symbol, side, qty, client_order_id, dry_run`. Records **order intent** — does NOT record fill price, fill timestamp, commission, or partial-fill state. That's the gap this work closes.
- Backtest cost model (`quant/backtest/engine.py`): `slippage_bps=5.0`, `commission_bps=0.0`. Commission default is accurate for Alpaca paper; slippage is the primary thing to validate.
- `scripts/live_watch.py` is a real-time event monitor (equity/position/PnL deltas → stdout for Claude Monitor tool). Does NOT do per-fill reconciliation. Complementary, not overlapping.
- `client_order_id` format from `quant/execution/orders.py:make_client_order_id`: `{strategy}-{YYYYMMDD}-{SYMBOL}-{8hex}` — clean 1:1 join key between local `trades.parquet` and Alpaca's order history.
- Rebalance cron fires Tue 15:55 ET (5 min before close), so the signal target was built from the *prior* trading day's close. Use NYSE calendar (already a project dep) to compute "prior trading day."

## §1 — Purpose & boundaries

A new on-demand script, `scripts/reconcile_live.py`, that pulls Alpaca fill data for a date range and produces a dated Markdown report comparing actual fills against backtest cost-model assumptions. The report covers slippage, timing, fidelity, and per-symbol breakdown for every order in `data/live/trades.parquet`. Reports live in `docs/live-recon/` and are committed by hand as a permanent audit trail of model-vs-reality drift.

**In scope:** fill outcome analysis for orders submitted via the project's own rebalance pipeline.

**Out of scope:**
- Real-time alerting (already covered by `scripts/live_watch.py`)
- Automated cron execution (Z is on-demand by design)
- Feeding findings back into strategy code (that's B, a separate later spec once enough reports accumulate)

## §2 — Architecture & components

Three units, each one file, each independently testable:

1. **`quant/live/recon.py`** *(new, ~150 LOC)* — pure logic. Takes two DataFrames (local `trades.parquet` slice + Alpaca `Order` list) plus a bar fetcher, returns a `ReconciliationReport` dataclass with per-fill rows and aggregate stats. No I/O, no Alpaca calls, no file writes.
2. **`quant/live/recon_render.py`** *(new, ~100 LOC)* — pure formatter. Takes a `ReconciliationReport`, returns a Markdown string. Sectioned: header, slippage table, timing table, fidelity table, per-symbol breakdown.
3. **`scripts/reconcile_live.py`** *(new, ~60 LOC)* — orchestrator. Parses `--since/--until` CLI args, loads `trades.parquet`, calls `AlpacaClient.list_orders()`, fetches bars via `quant/data/bars.py`, hands off to `recon.py` then `recon_render.py`, writes to `docs/live-recon/YYYY-MM-DD.md`. The only I/O layer.

**Why this split:** the logic is testable without mocking Alpaca; the renderer can be reused later if recon findings ever surface in tear-sheets (deferred Q6). The script uses the existing `quant/execution/alpaca.py` SDK wrapper, not the MCP (MCP is for interactive sessions; this is headless).

## §3 — Data flow

```
trades.parquet (intent)              Alpaca orders API (outcome)
        │                                       │
        │  client_order_id is the join key      │
        └──────────────┬────────────────────────┘
                       ▼
              merge on client_order_id
                       │
                       ▼
        for each fill: fetch bar close on prior_trading_day(submission_date)
                       │
                       ▼
        compute: slippage_bps, fill_lag_seconds, fill_ratio
                       │
                       ▼
        ReconciliationReport (per-fill rows + aggregates)
                       │
                       ▼
        Markdown render → docs/live-recon/YYYY-MM-DD.md
```

**Signal-price rule:** the close at the **rebalance-target date itself** (the `date` column in `trades.parquet`). This matches what the strategies actually use: `asof_index(history, asof)` in `quant/strategies/_common.py` resolves to the bar at `asof` when present, falling back to T-1 only when the asof bar is unavailable. (An earlier draft of this spec assumed the strategy used T-1; the assumption was empirically wrong and produced inflated slippage values on the first smoke run. Corrected in commit landing this design.)

**Slippage definition:**
- Buy: `(actual_fill_price - signal_close) / signal_close × 1e4` bps (positive = paid more than expected)
- Sell: `(signal_close - actual_fill_price) / signal_close × 1e4` bps (positive = received less than expected)

**Modeled benchmark:** 5.0 bps from `BacktestConfig.slippage_bps`, loaded at runtime so it stays in sync if the default changes.

## §4 — Error handling

One bad row does not abort the report. Per-row failures surface in the fidelity section.

| Condition | Behavior |
|---|---|
| Order in `trades.parquet`, no matching Alpaca order | Row marked `missing`, included in fidelity section |
| Order rejected/canceled by Alpaca | Row marked `rejected`, fidelity section, slippage = N/A |
| Partial fill (filled_qty < submitted_qty) | Row marked `partial`, slippage computed on filled portion |
| Bar fetch fails for signal_date/symbol | Row marked `no_signal_price`, slippage = N/A |
| `trades.parquet` empty for date range | Script exits 0 with a single-line "no trades to reconcile" report |
| Alpaca API error | Script exits non-zero; report not written; preserves prior report if any |

## §5 — Testing

- **`tests/live/test_recon.py`** — unit tests on `quant/live/recon.py` with synthetic DataFrames. Covers: clean 1:1 match, missing Alpaca order, rejected order, partial fill, no signal price, mixed buy/sell, multi-strategy aggregation. No network.
- **`tests/live/test_recon_render.py`** — snapshot tests on the Markdown renderer using a fixed `ReconciliationReport` fixture.
- **No end-to-end test for `scripts/reconcile_live.py`** — thin orchestrator; consistent with how other `scripts/` wrappers are handled.
- Target: ~15 new tests. Fits the existing pytest/ruff/mypy strict pipeline with no config changes.

## §6 — File layout, runtime, resolved questions

```
quant/live/recon.py              (new, ~150 LOC)
quant/live/recon_render.py       (new, ~100 LOC)
scripts/reconcile_live.py        (new, ~60 LOC)
tests/live/test_recon.py         (new, ~12 tests)
tests/live/test_recon_render.py  (new, ~3 snapshot tests)
docs/live-recon/                 (new dir, .gitkeep)
docs/live-recon/YYYY-MM-DD.md    (written by script, committed by hand)
```

**Q1 (signal-price source):** Re-fetch bars at recon time using `quant/data/bars.py`. No schema change to `trades.parquet`. Use prior trading day's close per NYSE calendar.

**Q2 (state management):** Stateless / idempotent. CLI: `python scripts/reconcile_live.py [--since YYYY-MM-DD] [--until YYYY-MM-DD]`. Defaults: `--until` = today, `--since` = 7 days before `--until`. Re-running with the same window overwrites the report — deterministic given input. No high-watermark or "last reconciled" tracking — keeps the script truly stateless.

**Q3 (file naming):** `docs/live-recon/YYYY-MM-DD.md` keyed on the `--until` date.

**Q4 (commit policy):** Script writes to disk only. Does not auto-commit. User reviews and `git add` by hand.

**Q5 (MCP vs SDK):** Existing `AlpacaClient` SDK wrapper for the script (headless). MCP reserved for interactive Claude sessions.

**Q6 (tear-sheet integration):** Deferred. Keep `docs/live-recon/` separate for now. The §2 renderer split makes future integration cheap (~20 lines in `quant/cli.py`).

**Runtime:** `uv run python scripts/reconcile_live.py`. Expected wall-clock <5s for any realistic window — small DataFrames, ≤30 bar fetches.
