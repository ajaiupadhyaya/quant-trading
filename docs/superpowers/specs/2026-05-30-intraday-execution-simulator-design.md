# Intraday Execution Simulator + Backtester — design spec

**Date:** 2026-05-30
**Status:** approved (user approved the design), pending implementation plan.
**Program:** sub-project **B** of the intraday-equities system (A→B→C→D→E). Depends on **A**
(the data layer, built). See [[project-quant-trading-intraday]] / sub-project A spec for context.

## Motivation

At seconds-to-minutes holding periods, returns are dominated by **execution** — spread, latency,
slippage, impact — not by the signal. A backtester that fills naively will manufacture profits that
evaporate live. B is the **honest-cost validator**: an event-driven simulator that replays the data
layer's events through a strategy, models fills realistically against the 1-second NBBO with
latency and the existing cost models, and produces returns the charter validation battery can judge.
It also defines the **`IntradayStrategy` protocol** that the future live engine (D) will reuse — so a
strategy runs identically in backtest and live (the execution-side analogue of the data layer's
`replay()`==`subscribe()` guarantee).

## Locked design decisions

- **Fill fidelity: hybrid** — marketable orders cross the spread; limit orders fill only when the 1s
  NBBO trades through them, with no passive-queue credit. (Sub-second limit-queue modeling is out of
  scope; it would need raw quotes the data layer does not store.)
- **Reuse existing cost models** from `quant/backtest/`: `market_impact_bps` + `trailing_dollar_adv`
  (`impact.py`), `financing_charge` (`financing.py`), and `metrics.py` for stats.
- **New event-driven Strategy protocol** — NOT the daily `quant/strategies/base.Strategy` (which is
  pull-based `target_positions(asof)`); the paradigms differ.
- **Charter applies:** realistic costs, no lookahead, reproducibility.

## Architecture — `quant/intraday/` (sibling to `data/`)

| Module | Responsibility |
|---|---|
| `strategy.py` | `IntradayStrategy` Protocol, `Order`/`OrderType`/`Side`, `StrategyContext` (read-only, ≤ current ts). |
| `sim/fills.py` | Fill model (marketable + limit-vs-NBBO) + cost application. Pure. |
| `sim/portfolio.py` | Position / cash / realized+unrealized P&L accounting, mark-to-mid. Pure. |
| `sim/engine.py` | Event loop: `replay()` → strategy `on_event` → latency queue → fills → portfolio. |
| `sim/result.py` | `BacktestResult`: intraday equity curve, daily-resampled returns, fills, trades, cost breakdown, metrics. |

### The Strategy protocol (shared with D)

```python
class Side(Enum): BUY = "buy"; SELL = "sell"
class OrderType(Enum): MARKET = "market"; LIMIT = "limit"

@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    qty: int
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None   # required iff type == LIMIT

class StrategyContext(Protocol):
    """Read-only market+account view, exposing ONLY data up to the current event ts."""
    def position(self, symbol: str) -> int: ...
    def cash(self) -> float: ...
    def nbbo(self, symbol: str) -> "QuoteBar | None": ...   # latest 1s NBBO seen
    def now(self) -> datetime: ...

class IntradayStrategy(Protocol):
    def on_event(self, event: Event, ctx: StrategyContext) -> list[Order]: ...
    def on_start(self, ctx: StrategyContext) -> None: ...   # optional, default no-op
    def on_end(self, ctx: StrategyContext) -> None: ...      # optional, default no-op
```

`Event` = `Trade | QuoteBar | Bar` from `quant.intraday.data.events`. The same `on_event` is called
in backtest and live → the execution-side anti-skew guarantee.

### Event loop + latency (`engine.py`)

1. Iterate `store.replay(symbols, start, end, datasets=...)` in strict ts order.
2. Maintain a live market view (latest NBBO + last trade per symbol) updated as events arrive.
3. Call `strategy.on_event(event, ctx)`; returned orders enter a **pending queue keyed by effective
   time = event.ts + latency** (configurable `latency`, default 250 ms).
4. Before processing each new event, drain any pending orders whose effective time ≤ that event's ts
   and attempt fills against the **prevailing** NBBO/last-trade at that time (never the future).
5. Fills update the portfolio; the equity curve is marked-to-mid on a configurable cadence
   (default per-minute) and at each fill.

### Fill model (`fills.py`, pure)

- **MARKET/IOC:** fill at effective time at the **far touch** (buy@ask, sell@bid) from the prevailing
  1s NBBO; add `market_impact_bps(notional, trailing_dollar_adv, impact_coef_bps)` to the fill price;
  add commission (configurable: per-share `commission_per_share` or `commission_bps`). Spread is paid
  implicitly via the far touch.
- **LIMIT:** rests in the book; fills only when a subsequent event shows the NBBO **trading through**
  the limit (a `Trade` print at/through `limit_price`, or the opposite touch crossing it), at
  `limit_price` (or better), with NO passive-queue credit. Uncrossed limits expire at end-of-run (or
  on explicit cancel — cancel API deferred to D; B expires at run end).
- **Overnight holds:** a position carried past the session close accrues `financing_charge`
  (borrow/margin) for the days held. Strategies may flatten by close; the model does not require it.

### Portfolio + result

`portfolio.py`: signed position, average cost, cash, realized + unrealized P&L, marked-to-mid per
event. `result.py` `BacktestResult` exposes: intraday equity curve, **daily-resampled OOS returns**
(so the existing charter stats and — in sub-project C — DSR/PSR/bootstrap/walk-forward operate on
daily returns exactly as the daily system does), the fills/trades log, a cost breakdown (spread vs
commission vs impact vs financing), and summary metrics via `quant/backtest/metrics.py` (passing the
correct `periods_per_year` for the chosen return frequency).

## Fail-safe / no-lookahead invariants (charter)

1. `StrategyContext` exposes only data with ts ≤ the current event — the strategy cannot read the
   future.
2. Latency strictly delays order effect; an order can never fill at a time before its effective time.
3. Deterministic: same fixture (+ seed, if any partial-fill randomness is enabled — default off) →
   identical `BacktestResult`.
4. Costs are charged on every fill; a "zero-cost" run is impossible by construction.

## Testing

- **`fills.py`:** marketable buy → ask + impact + commission; marketable sell symmetric; impact
  scales with order size; limit buy fills only when a later print trades through it (and not before);
  uncrossed limit expires; financing applied to an overnight hold.
- **`portfolio.py`:** position/cash/P&L accounting; round-trip P&L = Σ fills − costs; mark-to-mid.
- **`engine.py`:** latency (an order emitted at T fills against the T+latency NBBO, not the T NBBO);
  **no-lookahead** (a probe strategy cannot observe a future event); a deterministic toy strategy on a
  synthetic fixture yields the exact expected equity curve and cost breakdown.
- **determinism:** same fixture → identical result.
- All unit tests fixture/synthetic — no network, no Alpaca subscription required.

## Out of scope

- Real strategies (C), live engine / order routing (D), ops (E).
- Sub-second limit-queue modeling; partial-fill-by-displayed-size (impact captures size penalty).
- Multi-account / portfolio-margin nuance beyond the existing financing model.
