# Intraday Live Loop — Design Spec

**Date:** 2026-06-07
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project 0 ("the spine") of the intraday/60s showcase track.

---

## Context

`quant-trading` today is a disciplined, **daily**-rebalanced multi-strategy system
that trades paper money on Alpaca via a single live rebalance at 15:55 ET, plus a
continuous read-only monitoring/halt daemon. It already has an **intraday
*simulation* engine** (`quant/intraday/{data,sim,strategy.py,cli.py}`) that runs on
historical replay — but **no live intraday path**.

The owner's goal for this track is a **portfolio / learning showcase**: demonstrate
mastery across the hard techniques named in the original brief (intraday
infrastructure, market-making, optimal execution, RL execution, DL alpha). Because
it is a showcase rather than a real-money system, marginal/thin edge is acceptable;
breadth and technical correctness are the win.

The full intraday track decomposes into independent sub-projects, each with its own
spec → plan → build cycle:

- **0 — Spine (this spec):** continuous 60s loop + live intraday feed + ring-fenced
  sleeve + one proof-of-life strategy. Everything below depends on it.
- **A — Optimal Execution** (Almgren–Chriss TWAP/VWAP)
- **B — Market Making** (Avellaneda–Stoikov)
- **C — RL Execution Agent** (PPO/SAC, trained in the simulator)
- **D — DL Alpha** (LSTM/Transformer intraday signals)
- **E — NLP sentiment expansion**

This spec covers **only Sub-project 0**.

---

## Decisions locked in brainstorming

1. **Acting model:** the loop **acts on live paper from day one** (no shadow-only
   phase).
2. **Relationship to the daily system:** **additive**. The daily system (15:55 ET
   rebalance, 4 live strategies) is **untouched**. The 60s loop is a new layer
   trading a **separate, ring-fenced capital sleeve**.
3. **Proof-of-life strategy:** **intraday mean-reversion on liquid ETFs** (flat by
   close). This is honest work for the loop while B/C/D are built; it is *not* the
   showcase centerpiece.
4. **Guardrail profile:** **tight & safe** (numbers in §4).
5. **Training vs acting (critical):** training/retuning happens **offline** on
   historical data + the simulator (existing nightly/weekly jobs). The live loop
   **never learns from live fills** — it reads promoted params and acts, and feeds
   *drift observations* back to inform the next offline retune.

---

## 1. Scope & boundary

The spine is **infrastructure + one proof-of-life strategy**. In scope:

- A continuous **60s tick loop**, supervised for 24/7 uptime, that acts only during
  the equities session and idles (heartbeat-only) otherwise.
- A **live intraday data feed** from Alpaca (quotes / minute bars) with reconnect.
- A **ring-fenced intraday sleeve** with its own internal ledger, P&L, and
  guardrails inside the single Alpaca paper account.
- **Intraday mean-reversion** on a disjoint liquid-ETF universe as the first
  workload.
- A **drift/monitoring** hook connecting live behavior to the existing governance
  drift machinery.

**Explicitly out of scope** (future sub-projects): market-making, optimal-execution
slicing, RL agent, DL alpha, NLP, any change to the daily system, WebSocket
streaming (REST polling is sufficient at 60s), multi-account/sub-account separation.

---

## 2. Architecture

New subpackage `quant/intraday/live/`:

| File | Purpose | Depends on |
|---|---|---|
| `loop.py` | The continuous tick engine (the spine). Orchestrates the lifecycle in §3. | feed, sleeve, guardrails, strategy, `execution/alpaca`, `live/bookkeeping` |
| `feed.py` | Live intraday data feed (Alpaca quotes/minute bars) + reconnect/backoff. Normalizes to the shape the strategy consumes. | `execution/alpaca`, `quant/intraday/data` |
| `sleeve.py` | Sleeve accounting: an **internal ledger** keyed by a dedicated `client_order_id` namespace; tracks sleeve positions + realized/unrealized P&L independently of the Alpaca aggregate. | `live/bookkeeping`, `util` |
| `guardrails.py` | Notional cap, per-trade cap, trade-count budget, daily-loss kill-switch, flat-by-close. Triggers sleeve-local halt. | `governance/`, `util/calendar` |
| `strategy.py` | Tick-based intraday strategy interface + the mean-reversion proof-of-life implementation. Reuses/extends `quant/intraday/strategy.py` where the interface fits. | `quant/intraday/strategy.py` |

**Reuses (no changes to public behavior):** `execution/alpaca.py` (order submission),
`governance/` halt machinery, `live/bookkeeping.py` (journal/equity persistence),
`monitor/daemon.py` (the long-running supervised-daemon pattern), `util/calendar`
(trading-session checks), `data/universe.py` (universe registry).

### Ring-fencing the sleeve inside one Alpaca account

Alpaca paper is a single account where positions net. The sleeve is isolated two
ways, in priority order:

1. **Disjoint universe (primary):** the sleeve trades ETFs the daily strategies do
   **not** hold, so positions never collide. Daily universe is
   `SPY, TLT, IEF, GLD, DBC, VNQ, EFA, EEM`; sleeve universe is **`QQQ, IWM, DIA`**
   (all ultra-liquid, good intraday mean-reversion vehicles, zero overlap). Room to
   add sector SPDRs (XLK/XLF/…) later, still disjoint.
2. **Internal ledger (always):** the sleeve tracks its **own** positions and P&L
   from its **own** `client_order_id`-tagged fills — never from the Alpaca
   aggregate. This mirrors the daily rebalance, which already reconciles against its
   own snapshot, not the Alpaca net. If a future overlap is ever unavoidable, the
   internal ledger remains correct; the disjoint-universe rule simply keeps the
   Alpaca-level view legible too.

---

## 3. Tick lifecycle (every 60s)

On each tick:

1. **Session check** — if the equities session is closed, emit heartbeat and idle.
2. **Guardrails first** — evaluate kill-switches *before* acting. If the sleeve's
   daily-loss threshold is breached or a sleeve-halt is set → **flatten all sleeve
   positions and remain halted** (resume is manual). If within the flat-by-close
   window → flatten and stop opening.
3. **Pull live data** for the sleeve universe via `feed.py` (latest quotes / minute
   bars). On feed failure → skip this tick's *new* actions, keep heartbeat, retry
   with backoff (never blind-trade on stale data).
4. **Strategy** → target sleeve weights. Mean-reversion: when a name deviates beyond
   an entry z-band from its short intraday VWAP/rolling mean, fade it; exit toward
   the mean or at the exit band.
5. **Sizing + pretrade gates** — apply per-trade cap, notional cap, and remaining
   trade-count budget. Clamp/skip orders that would breach any limit.
6. **Reconcile** target vs the internal sleeve ledger → compute order deltas.
7. **Submit** deltas to Alpaca with the sleeve `client_order_id` namespace.
8. **Record** the tick: sleeve positions, intended actions, submitted orders, fills,
   and sleeve P&L → journal (parquet/jsonl via `live/bookkeeping`).
9. **Sleep** to the next 60s boundary.

**Flat-by-close:** ~15 minutes before the session close, flatten all sleeve
positions. The sleeve carries **no overnight risk**.

---

## 4. Sleeve & guardrails ("tight & safe")

| Control | Value | Behavior on breach |
|---|---|---|
| Sleeve size | `min(10% of paper equity, $10,000)` hard notional cap | New opens that would exceed the cap are clamped/skipped |
| Per-trade cap | ~$2,000 notional | Order clamped down to the cap |
| Trade budget | ~20 round-trips/day | Further opens skipped once exhausted (exits still allowed) |
| Daily-loss kill-switch | sleeve down ~1.5% of its allocation on the day | **Auto-flatten + sleeve-halt** (manual resume) |
| Flat-by-close | ~15 min before close | Flatten all, stop opening |

All thresholds live in a config/TOML artifact (no magic numbers per the Charter),
overridable per the existing config pattern. Exact values confirmable at plan time.

**Sleeve-local halt vs global halt:** the daily-loss kill-switch sets a
**sleeve-scoped** halt that stops only the intraday loop. It must **never** freeze
the daily system. The existing global `governance halt` still applies on top (a
global halt also stops the sleeve), but a sleeve halt does not propagate upward.

---

## 5. Training/retuning ↔ acting (the two pipelines)

**Offline (training/retuning) — unchanged machinery:**
- The mean-reversion params (lookback window, entry/exit z-bands, universe) are
  **fit and validated on historical intraday data** through the existing
  backtest/validation discipline (walk-forward, DSR/PSR, cost sweep) + the intraday
  sim engine.
- The existing nightly backtest / weekly grid-search / weekly validation+governance
  jobs **retune** them. A promoted parameter set is written to a governance/config
  artifact.

**Online (acting + monitoring) — this spec:**
- The loop **reads the promoted params** and acts. It does **not** learn from live
  fills.
- Every tick is journaled. A daily intraday-recon compares live sleeve P&L against
  the strategy's backtested expectation; the existing `governance drift` surfaces
  divergence.

**The connection:** drift observations from the online pipeline are *inputs* to the
next offline retune — not a live-learning loop. (The RL agent in sub-project C is the
only component that learns from interaction, and it does so **in the simulator**, not
from live paper fills.)

---

## 6. Infrastructure (M4, 24/7)

- A `launchd` plist runs `quant intraday live run` as a supervised daemon
  (`KeepAlive=true`, auto-restart with backoff) — mirroring how the monitoring
  daemon is hosted. The **process is always up** ("24/7"); it only *acts* in-session
  and idles otherwise.
- **Crash recovery:** on startup, rebuild sleeve state from the internal ledger,
  reconcile against current Alpaca sleeve positions, then resume. If reconciliation
  finds a mismatch beyond tolerance → start halted and require manual review.
- **New CLI surface** (`quant intraday live …`):
  - `run` — start the loop (foreground/daemon)
  - `status` — heartbeat, sleeve P&L, open positions, halt state, trade budget used
  - `halt` / `resume` — sleeve-local kill-switch (manual)
  - `flat` — manual flatten of all sleeve positions
- **Cadence:** 60s, configurable. Out of session → heartbeat only.

---

## 7. Testing

- **Unit:**
  - Sleeve ledger accounting (position + realized/unrealized P&L math) from a fill
    stream.
  - Every guardrail trigger: daily-loss halt → auto-flatten, trade-count exhaustion,
    per-trade clamp, notional-cap clamp, flat-by-close window.
  - Mean-reversion signal generation on fixture data (entry/exit band logic).
  - Feed parsing + reconnect/backoff (mocked Alpaca).
- **Integration:** replay a historical intraday session through the full loop under a
  **fake clock** (mocked Alpaca + feed). Assert: it opens/closes positions, respects
  every cap, flattens by close, and halts when the loss threshold is hit. Leverages
  the existing sim engine for the data path.
- **Property:** deterministic given fixed data; the sleeve **never** exceeds notional
  or trade-count caps under any generated input sequence.

---

## 8. Charter compliance

- **No lookahead:** strategy uses only data available at each tick; offline fitting is
  PIT/walk-forward.
- **Realistic execution:** orders go through the real Alpaca paper path; sleeve P&L is
  measured from actual fills, not theoretical mid.
- **Robust validation:** params are promoted only after the existing offline
  validation battery, not chosen live.
- **Overfitting guard:** the proof-of-life strategy is intentionally simple
  (few params); DSR/PSR gating applies to its offline promotion.
- **Reproducibility:** config-driven thresholds, journaled ticks, deterministic
  tests, git-committed artifacts.

---

## 9. Success criteria

- The daemon runs continuously on the M4 across a reboot (launchd KeepAlive).
- During a live session it places sleeve orders on Alpaca paper, tagged and tracked
  in the internal ledger, and the daily system is provably unaffected.
- All guardrails fire correctly (verified in integration tests and observable via
  `status`): caps respected, flat by close, auto-halt on loss threshold.
- A day's ticks are journaled and a drift comparison against the backtest is
  produced.
- The daily system's behavior and artifacts are byte-unchanged by the loop's
  presence.

---

## 10. Out of scope / deferred

Market-making, optimal-execution slicing, RL agent, DL alpha, NLP, WebSocket
streaming, crypto sleeve (would extend "act in-session" to literal 24/7 — revisit
after the equities spine is proven), and any modification to the daily system. Each
future sub-project gets its own spec.
