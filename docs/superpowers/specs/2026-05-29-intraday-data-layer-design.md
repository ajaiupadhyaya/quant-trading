# Intraday Data Layer — design spec

**Date:** 2026-05-29
**Status:** approved (user delegated final design choices: "do whatever is best"), pending implementation plan.
**Program:** sub-project **A** of the intraday-equities trading system (A→B→C→D→E). This spec
covers **A only**. See "Program context" below.

## Program context

The user is extending the `quant-trading` project from a daily-batch paper-trading system into a
genuinely autonomous, always-on **intraday** equities system. Decisions locked during brainstorming:

- **Alpha cadence:** intraday, **seconds-to-minutes** holding periods (realistic edge expected at the
  ~1–5 min end; true sub-second HFT is explicitly out of scope — not feasible on retail infra).
- **Asset class:** US equities / ETFs. Paper account (no real money without explicit user request).
  Note for any future live decision: seconds-to-minutes = day-trading → the PDT rule walls live
  equities under $25k. Irrelevant on paper.
- **Data:** Alpaca **full SIP** feed ("Algo Trader Plus", ~$99/mo) — realtime + historical.
- **Hosting:** small **cloud VPS in us-east-1** runs the live engine; **Mac Mini (M4, 16GB)** is the
  dev/research box; **Raspberry Pi 4B** is the heartbeat/watchdog (built in sub-project E).
- **Codebase:** a parallel in-repo package `quant/intraday/`, reusing the charter's validation
  statistics (Deflated Sharpe, PSR, bootstrap, walk-forward) but with its own event-driven engine.

The five sub-projects, built in dependency order:

```
A. Intraday Data Layer  ── THIS SPEC ── trustworthy intraday data (historical + realtime)
B. Event-Driven Backtest + Execution Simulator   (depends on A)
C. Strategy + Validation Pipeline                (depends on A, B)
D. Live Trading Engine                           (depends on A, shares strategy code with C)
E. Ops & Resilience (VPS deploy, Pi watchdog, dashboard, alerts, kill-switch)  (wraps D)
```

The charter (`docs/CHARTER.md`) governs all of it: no lookahead / point-in-time data, realistic
execution costs, robust out-of-sample validation, overfitting guard, full reproducibility.

## Motivation

Everything downstream — honest backtests, live signals, risk — depends on a single trustworthy
source of intraday truth. The classic failure mode is **train/serve skew**: the backtester and the
live engine silently read data through different code paths, so a strategy that "worked" in
backtest behaves differently live. The data layer's job is to make that impossible by serving both
consumers **the same event stream through the same interface**.

## Locked scope

- **Universe:** ~100 most-liquid US names (large-caps + major ETFs: SPY/QQQ/IWM/etc.). Intraday edge
  requires liquidity; the illiquid tail has no tradeable edge net of spread. Point-in-time
  membership (no survivorship bias).
- **Granularity / substrate:** **full trade tape + 1-second NBBO bars** (best bid/ask/mid/spread
  sampled each second), full history (~8 yrs, ~1 TB). 1-second NBBO captures the spread — the
  dominant intraday cost — at a resolution ample for seconds-to-minutes execution modeling. Raw
  sub-second quotes are **not** stored standing; they can be pulled on-demand for a specific
  symbol/window if a future strategy genuinely needs sub-second queue dynamics.
- **Derived convenience:** 1-minute OHLCV bars (from trades) for fast strategy iteration.

## Storage approach

**Partitioned Parquet + DuckDB/Polars** (an embedded "mini-lakehouse"). No DB server to run or
secure; portable across Mac Mini ↔ VPS ↔ cloud; columnar + compressed; DuckDB queries Parquet
directly, including over object storage.

Layout (partitioned by `symbol/date`):
```
<root>/trades/symbol=AAPL/date=2023-06-01.parquet          raw trades (ts, price, size, exch, conds)
<root>/quote_bars_1s/symbol=AAPL/date=2023-06-01.parquet   1s NBBO (ts, bid, ask, bid_sz, ask_sz, mid, spread)
<root>/minute_bars/symbol=AAPL/date=2023-06-01.parquet     derived 1m OHLCV (+ vwap, trade_count)
<root>/adjustments/symbol=AAPL.parquet                     split/dividend factor table
<root>/_meta/universe.parquet                              PIT universe membership
```

**Canonical store:** Cloudflare R2 or Backblaze B2 (cheap; R2 has zero egress so both machines pull
freely), with a local SSD cache per machine. Bootstrap simply: external SSD on the Mac Mini as
canonical + an R2/B2 mirror; graduate to object-store-canonical later. Storage path is configurable.

Alternatives considered and rejected for now: QuestDB/TimescaleDB (a server to operate; revisit only
if realtime ingest rate outgrows embedded), ClickHouse (heaviest ops; over-engineered at this scale).

## Architecture — `quant/intraday/data/`

| Module | Responsibility |
|---|---|
| `events.py` | Shared event types: `Trade`, `Quote`/`QuoteBar`, `Bar`. The single vocabulary `replay()` and `subscribe()` both emit. |
| `config.py` | Universe list, storage roots, SIP settings, session calendar source. |
| `universe.py` | ~100 liquid names + **point-in-time** membership (listings/delistings/symbol changes). |
| `backfill.py` | Historical trades+quotes ingester from Alpaca SIP REST. Idempotent, **resumable** (per symbol/date checkpoints), rate-limit-aware. |
| `aggregate.py` | quotes → 1s NBBO bars; trades → 1m OHLCV. Pure, unit-testable. |
| `adjustments.py` | Split/dividend factor table; applied at **read time** (raw never rewritten), parameterized by `as_of`. |
| `store.py` | `MarketDataStore`: `get_trades/get_quote_bars/get_minute_bars(...)`, plus `replay()` and `subscribe()`. The single read interface. |
| `stream.py` | Async SIP websocket ingester + in-memory rolling buffer + periodic flush to today's partition. Auto-reconnect + gap backfill. |
| `quality.py` | Gap detection, market-calendar alignment (halts/half-days/early closes), dedup, bad-tick filter, and a `doctor` health check. |

CLI surface (mirrors the existing `quant` tool): `quant intraday data backfill | refresh | status | doctor`.

### The central interface (load-bearing)

```python
# events.py — identical shapes in backtest and live
@dataclass(frozen=True)
class Trade:    ts: datetime; symbol: str; price: float; size: int; ...
@dataclass(frozen=True)
class QuoteBar: ts: datetime; symbol: str; bid: float; ask: float; bid_sz: int; ask_sz: int  # 1s NBBO
@dataclass(frozen=True)
class Bar:      ts: datetime; symbol: str; open: float; high: float; low: float; close: float; volume: int  # 1m

# store.py
class MarketDataStore:
    def replay(self, symbols: list[str], start, end, *, as_of=None) -> Iterator[Trade | QuoteBar | Bar]:
        """Backtester (B): events in STRICT timestamp order across symbols, PIT-adjusted as_of."""
    def subscribe(self, symbols: list[str]) -> Iterator[Trade | QuoteBar | Bar]:
        """Live engine (D): the SAME event shapes from the realtime buffer."""
    def freshness(self) -> "Freshness":
        """Health signal D uses to fail-closed (halt trading) on stale/gapped data."""
```

A strategy consumes one event type and **cannot tell whether it is in a backtest or live.** This is
the structural guarantee against train/serve skew, and the reason the data layer is built first.

## Data flow

- **Backfill:** Alpaca REST → `backfill` → raw trades + quotes → `aggregate` → Parquet partitions →
  object store + local cache.
- **Realtime:** Alpaca WS → `stream` → in-memory rolling buffer (+ periodic flush to today's
  partition) → `subscribe()`.
- **Read (backtest):** DuckDB/Polars over Parquet → `replay()` → strategy.
- **Read (live):** rolling buffer → `subscribe()` → strategy. The VPS holds only a recent **hot
  window** (a few days' lookback), not the full TB.

## Point-in-time correctness (charter principle #1)

Store **raw, unadjusted** prices + a separate split/dividend **factor table**. Apply adjustments at
read time, parameterized by an `as_of` date, so a backtest reading "as of date D" sees only
corporate actions known by D. History is never mutated; PIT reconstruction is exact.

## Resilience / error handling

- **Backfill:** resumable checkpoints; exponential backoff on rate limits; row-count verification vs
  expected session length; corrupt-partition quarantine (never half-write a partition).
- **Realtime:** auto-reconnect with backoff; on reconnect, **REST-backfill the gap**; staleness
  heartbeat (no ticks for N sec during market hours → flag + surface via `freshness()`). A stream
  gap must never silently corrupt state.
- **Fail-closed hook:** `freshness()` lets the live engine (D, later) halt trading on stale/gapped
  data — the intraday analogue of the existing daily `quant doctor` / reconciliation guards.

## Testing

- **Unit:** aggregation (synthetic ticks → 1s NBBO and 1m OHLCV); adjustment math; **PIT
  correctness** (a read as-of D must not see a split announced after D); gap detection; bad-tick
  filter; session-calendar edge cases (half-days, halts).
- **Anti-skew golden test (the keystone):** `replay()` and `subscribe()` yield **identical** event
  sequences over a recorded fixture. This invariant is the whole point of the layer.
- **Integration:** backfill one symbol/day against a **recorded Alpaca fixture**; assert partitions
  written, queryable, row counts sane.
- **No network in unit tests** (record/replay Alpaca responses); seeded + reproducible (charter #5).

## Prerequisites (user actions, external)

- Subscribe to Alpaca **Algo Trader Plus** (full SIP) and provide API keys via the existing secret
  mechanism. Code + unit tests are built against fixtures and need no subscription; **backfill and
  integration tests against live data require it.**
- Provision storage: an external SSD on the Mac Mini and/or an R2/B2 bucket. (VPS provisioning is
  sub-project E.)

## Out of scope (this spec)

- The execution simulator / backtester (B), strategy research (C), live engine (D), ops (E).
- Storing standing raw sub-second quotes (on-demand pulls only).
- Non-equity asset classes.
- Any change to the existing daily-batch system or its governance.
```
