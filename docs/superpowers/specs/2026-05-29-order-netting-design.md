# Per-symbol order netting — design spec

**Date:** 2026-05-29
**Status:** approved (user chose "build the netting fix"), pending plan.

## Motivation

`run_rebalance` submits each strategy's orders **inline and independently**. The
account is a single shared Alpaca account, so when a live strategy BUYS a symbol
in the same run that an orphan wind-down SELLS it, Alpaca rejects the second
(opposing) order ("cannot sell while a buy is open" / wash-trade). Observed
live: defensive-etf re-sizes DBC/EEM/GLD every run; whenever its per-run tweak
is a buy, the concurrent orphan sell of that symbol is rejected, leaving a
persistent ~$97k DBC orphan excess that never clears.

Fix: **net all intended orders per symbol before submitting** — one net order
per symbol = the account's desired delta. This eliminates opposing-order
rejections entirely (live-vs-orphan and, generally, any two strategies), and
realizes "adopt under governance": a live strategy buying a symbol an orphan
holds simply *absorbs* those shares via a smaller (or zero) net order instead of
a doomed sell.

## Core idea

Per-strategy snapshots remain the unit of **intent** (each strategy still writes
its own target/remaining snapshot, so `sum(snapshots)` = desired account and
reconciliation stays consistent). Only **order submission** changes from
"per-strategy inline" to "collect all intents → net per symbol → submit net".

## Architecture

### New pure unit — `quant/execution/netting.py`

```python
def net_orders(orders: list[OrderTemplate]) -> list[OrderTemplate]:
    """Collapse many per-strategy orders into one net order per symbol.

    Signed qty per symbol: BUY = +qty, SELL = -qty. Net > 0 -> one BUY of |net|;
    net < 0 -> one SELL of |net|; net == 0 -> no order (fully offsetting).
    The net order's strategy_slug is the slug contributing the largest |qty| to
    that symbol (ties broken alphabetically) — attribution is for the trade log /
    client_order_id only; the per-strategy SNAPSHOTS carry the true intent.
    Deterministic ordering (sorted by symbol) for reproducible client_order_ids.
    """
```

Pure, fully unit-testable. Examples:
- live BUY 11276 DBC + orphan SELL 3304 DBC → net BUY 7972 DBC (no opposing pair).
- orphan SELL 670 BAC only → net SELL 670 BAC (unchanged).
- live BUY 100 X + orphan SELL 100 X → net 0 → no order (perfect adoption).

### `run_rebalance` refactor (collect → net → submit)

1. **Live loop** (currently submits inline at ~408-425): instead, **append** its
   `reconcile` orders to a shared `intended: list[OrderTemplate]`, and keep the
   existing snapshot write (`write_strategy_positions(..., target)`, live-only)
   and `StrategyRebalanceOutcome` recording as **intent**. Remove only the inline
   `submit_order` loop.
2. **Orphan wind-down loop**: same — append the capped exit orders to `intended`;
   keep the orphan's intended-`remaining` snapshot write (live-only) and
   `WindDownOutcome`. Remove only the inline `submit_order`. (The per-exit
   "successful-exits-only" advance is no longer needed for the opposing-order
   case — netting removes that failure mode — so the orphan snapshot is just its
   intended `remaining`; the wind-down `winddown_orders` helper is unchanged, we
   simply collect its orders instead of submitting them.)
3. **Net + submit (new, after both loops, before `append_trades`):**
   ```
   net = net_orders(intended)
   for order in net:
       try:
           coid = client.submit_order(order, dry_run=dry_run)
       except Exception:
           log; continue   # rare now that opposing orders are netted away
       all_trade_rows.append({... net order ..., strategy=order.strategy_slug, coid, dry_run})
   ```

Snapshots are written inline as **intent** (= `sum(snapshots)` is the desired
account). Netting eliminates the *opposing-order* failure that the wind-down's
per-exit fail-safe was added for, so deferred/per-symbol snapshot reversion is
no longer warranted. A *rare* remaining net-submit failure (e.g. buying power)
leaves that symbol's snapshot optimistic → **next run's reconciliation refuses
to trade (fail-safe, visible) until resolved** — the conservative, already-built
guard. This keeps the refactor small and low-risk.

### Attribution detail

For a net order, `strategy_slug` = argmax over contributing strategies of the
absolute qty they put into that symbol. This means a net DBC buy (dominated by
defensive-etf's 11276) is attributed to defensive-etf; a net BAC sell (only
multi-factor) to multi-factor. The coid prefix therefore stays meaningful for
the trade log even though the order is netted.

## Fail-closed invariants (preserved)

1. **Never opens beyond intent:** netting only sums *intended* orders; it cannot
   create exposure no strategy intended. A net order moves the account toward
   `sum(snapshots)` = the live targets + orphan-zeros.
2. **No opposing orders ever submitted** (the whole point) — so the wash-trade
   rejection cannot occur.
3. **Dry-run:** still no real submit; snapshots not written.
4. **Rare net-submit failure → fail-safe via reconciliation:** snapshots are
   written as intent; a failed net order makes that symbol's expected book
   optimistic, so the next run's reconciliation guard refuses to trade (visible,
   conservative) until resolved. (Netting removes the common opposing-order
   failure; remaining failures are rare.)
5. **Reconciliation:** unchanged — it sums per-strategy snapshots
   (enabled + winddown), which still reflect intent.
6. Emergency-halt / market-open / risk-breaker guards unchanged (they gate
   before the loops).

## Testing

**`tests/execution/test_netting.py` (pure):**
- opposing buy+sell same symbol → net single order (or zero if equal);
- multiple buys + sells across strategies → correct signed net;
- non-overlapping symbols pass through unchanged;
- attribution = largest-|qty| contributor, ties alphabetical;
- empty input → [].

**`tests/live/test_rebalance.py` (integration):**
- a live BUY + orphan SELL of the SAME symbol now produces ONE net order (no
  opposing pair submitted); assert only one order per symbol reaches the stub
  client; the symbol nets correctly.
- the existing wind-down convergence + dry-run tests still pass (adapted to the
  netted submission: assert the stub received netted orders).
- partial-failure: a stub that fails one net symbol leaves all contributors'
  snapshots un-advanced for that symbol.

## Rollout / verification
- Full suite + mypy + ruff + format green.
- Live dry-run: confirm DBC shows ONE net order (defensive buy − orphan sell),
  no opposing pair; wind-down table still shows orphan intent.
- One real run: confirm the DBC orphan excess clears (net sell/reduced-buy) and
  the account converges to defensive-etf's exact targets.

## Out of scope
- Changing the per-strategy snapshot/attribution model itself (snapshots remain
  the intent unit; only submission is netted).
- The stray 1-share SPY orphan with no snapshot (separate tiny manual cleanup).
- Gate-calibration audit (separate deliverable).
