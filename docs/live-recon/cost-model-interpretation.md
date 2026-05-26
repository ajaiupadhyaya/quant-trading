# Cost-model interpretation note

**Date:** 2026-05-26
**Status:** Recalibration NOT applied; insufficient evidence and signed-metric ambiguity.

## Context

The first live reconciliation report (`docs/live-recon/2026-05-26.md`) showed
mean signed slippage of −9 to −12 bps across the four active strategies, with
the report header stating the backtest cost model is "~15 bps conservative".
That phrasing implied the engine's `BacktestConfig.slippage_bps = 5.0` could be
bumped down to reflect reality, increasing modeled returns.

That conclusion does not hold. This note documents why.

## What the recon actually measures

`quant/live/recon.py:_slippage_bps`:

```python
def _slippage_bps(side: str, signal_price: float, fill_price: float) -> float:
    if side == "buy":
        return (fill_price - signal_price) / signal_price * 1e4
    return (signal_price - fill_price) / signal_price * 1e4
```

`signal_price` is the close (or open) of the bar that produced the signal —
fixed by `quant/live/recon.py` at recon time. `fill_price` is the actual Alpaca
fill, which lands at the next regular-session open.

A signed delta of −10 bps therefore means "the fill happened at a price
10 bps better than the signal-time reference price" — not "execution cost the
strategy 10 bps".

## What the engine models

`quant/backtest/engine.py:apply_costs`:

```python
sign = +1.0 if side == "buy" else -1.0
fill_price = mid_price * (1.0 + sign * slip)
```

`slippage_bps` is a one-way cost: buys fill above mid, sells fill below mid,
both by the same bps. It models the half-spread plus market impact incurred at
execution time. It is, by construction, always a cost — never a credit.

## Why the two numbers are not comparable

The recon delta captures **price drift** between signal time and execution
time, not **execution cost** at execution time. With signal_price = close of
day T and fill_price = open of day T+1, an overnight market drift that happens
to favor the strategy will show up as negative recon slippage even if the
actual execution cost was the full 5 bps modeled.

For the 18-fill sample in 2026-05-26.md, the magnitude is also too small to
generalize: N=1–3 per symbol, JPM at −35 bps on a single trade, IEF/TLT/WMT/XOM
all positive. The strategy-level means inherit that variance.

## Decision

1. Leave `BacktestConfig.slippage_bps` at 5.0 bps.
2. Treat the signed recon delta as an execution-timing diagnostic, not a
   cost-model calibration signal.
3. Build the cost-model recalibration on a true execution-cost metric —
   `fill_price` vs the **fill-time** mid (or NBBO midpoint), not the
   signal-time close. That requires an intraday data source (Polygon, IEX,
   Alpaca minute-bars) and is a follow-up.
4. Until a true execution-cost metric exists, accumulate ≥4 weeks of fills
   before any cost-model adjustment; per-strategy N must be ≥30 for a per-
   strategy override.

## Follow-ups

- Add a `fill_vs_open` column to the recon parquet so the existing signed
  delta is preserved while a true execution-cost column can be added later.
- Replace the recon report's "vs modeled" column wording so it does not read
  as a calibration recommendation when the signal-vs-fill window can dominate
  the signed result.
- Once intraday bars are wired, implement `execution_cost_bps = (fill_price -
  mid_at_fill) / mid_at_fill * 1e4 * side_sign` and use that as the
  calibration target.
