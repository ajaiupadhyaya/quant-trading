# Governed orphan-position wind-down — design spec

**Date:** 2026-05-29
**Status:** approved (design), pending implementation plan.

## Motivation

The Alpaca paper account holds ~$660k of positions opened by strategies that
are now governance-quarantined (trend, multi-factor, risk-parity, momentum).
`run_rebalance` only iterates **governance-live** strategies (currently just
`defensive-etf-allocation`), reconciling each against *its own* per-strategy
snapshot in `data/live/strategy_positions.parquet`. Positions owned by
non-live strategies are therefore never reconciled or exited — they drift
unmanaged, and they silently inflate the account's risk.

This slice brings every such **orphan** position under active governance by
**winding it down to flat** in a controlled, fail-closed way. It is the honest
reading of "manage/exit under governance": the owning strategies have no
validated edge, so carrying their risk indefinitely is indefensible; managed
exit is the coherent outcome.

## Definitions

- **Orphan slug** = a slug that (a) has a non-empty latest snapshot in
  `strategy_positions.parquet`, (b) whose governance state is **not** `LIVE`,
  and (c) is present in `REGISTRY` (so its universe is resolvable). Derived
  fresh at each rebalance from governance state — never a stored flag — so a
  later QUARANTINED→LIVE flip (e.g. from a gate recalibration) automatically
  removes a slug from the wind-down set and returns it to normal management.

## Architecture

### New unit — `quant/live/winddown.py`

Pure/near-pure helpers, independently testable:

```python
def detect_orphans(data_dir: Path) -> list[str]:
    """Sorted slugs with a non-empty latest snapshot, governance state != LIVE,
    and present in REGISTRY. Returns [] if governance state is unavailable."""

def capped_qty(order_qty: int, adv_dollar: float, price: float,
               participation_fraction: float) -> int:
    """min(order_qty, floor(adv_dollar * participation_fraction / price)).
    Returns 0 when adv_dollar <= 0, price <= 0, or non-finite (cannot size)."""

def winddown_orders(
    slug: str, snapshot: dict[str, int], bars: pd.DataFrame, asof: date,
    participation_fraction: float, adv_window: int = 21,
) -> tuple[list[OrderTemplate], dict[str, float]]:
    """Exit-only orders that reduce `snapshot` toward flat, each capped at the
    ADV participation fraction. Returns (orders, reference_prices).

    Implementation: orders = reconcile(target={}, current=snapshot,
    strategy_slug=slug)  -> structurally flatten-only (sell longs / cover
    shorts; never opens). For each order, adv = trailing_dollar_adv(bars,
    order.symbol, pd.Timestamp(asof), adv_window); price = latest close;
    order.qty = capped_qty(order.qty, adv, price, participation_fraction).
    Orders capped to qty 0 are dropped. Symbols with no bars (adv 0) yield a
    0-qty drop and are returned in a skipped set for logging (never silently
    lost)."""
```

`winddown_orders` depends only on `reconcile` (`quant/execution/reconciler.py`),
`trailing_dollar_adv` (`quant/backtest/impact.py:43`), and `OrderTemplate`
(`quant/execution/orders.py`) — no engine/config import, so it's standalone and
unit-testable.

### Reconciliation change — `quant/live/safety.py`

`check_reconciliation` (safety.py:66-105) currently builds `expected =
_snapshot_aggregate(data_dir, enabled_slugs)` (line 79). During wind-down an
orphan still holds shares in Alpaca but is excluded from `expected`, so it reads
as `expected=0, actual=N` and (N > `tolerance_shares=1`) **fails the guard and
halts the entire rebalance — including the one live strategy.** Fix:

- Add `winddown_slugs: list[str] | None = None` to `check_reconciliation`.
- `expected = _snapshot_aggregate(data_dir, list(enabled_slugs) + list(winddown_slugs or []))`.
- `_snapshot_aggregate` (safety.py:56-63) already iterates whatever slugs it's
  given — no change needed there beyond receiving the union.

Effect: the orphan's current snapshot counts as "expected" at entry, so
reconciliation passes; after the wind-down zeroes the snapshot and the exits
fill, subsequent rebalances see expected→0 and actual→0, converging cleanly.
Tolerance stays 1 share.

### Orchestration change — `quant/live/rebalance.py`

1. **Detect orphans** immediately after `enabled` is finalized (~line 225,
   before the reconciliation guard at 256): `orphans = detect_orphans(settings.data_dir)`.
   (Detection must precede reconciliation so the slugs can be threaded in.)
2. **Thread into the guard:** `check_reconciliation(..., winddown_slugs=orphans)`
   at lines 256-260.
3. **Wind-down block** placed **after the main enabled-strategy loop (ends line
   424) and before `append_trades` (line 426)**, so orphan trade rows flush in
   the same `append_trades` call. For each orphan slug:
   - guard `slug in REGISTRY` (skip + log otherwise — mirrors the loop's 330-340
     pattern);
   - `bars = _bars_for(REGISTRY[slug], asof, history_days)` (its own universe —
     **must** fetch, else ADV is 0 and every exit no-ops);
   - `snapshot = last_strategy_positions(settings.data_dir, slug)`;
   - `orders, ref_prices = winddown_orders(slug, snapshot, bars, asof, participation_fraction)`;
   - if `not dry_run`: submit each order (collect `coid`), append rows to a
     parallel `orphan_trade_rows` list, **then** `write_strategy_positions(
     settings.data_dir, asof, slug, {})` to zero the snapshot atomically with
     submission;
   - record a `WindDownOutcome(slug, exited={symbol: qty}, orders, skipped, error)`
     on the report.
   - Flush `orphan_trade_rows` via the existing `append_trades` at 426-427.
4. **Report:** add `winddown_outcomes: list[WindDownOutcome]` to `RebalanceReport`
   so the CLI can render exits (observability).
5. **Parameter:** add `winddown_participation: float = 0.10` to `run_rebalance`;
   thread to `winddown_orders`. (10% of trailing dollar-ADV per name per
   rebalance — for the current all-liquid book the cap never binds, so orphans
   exit next rebalance; it protects against thin names generically.)

### CLI — `quant/cli.py`

- `quant rebalance` renders a "Wind-down" table (orphan slug, symbols exited,
  qty, capped?) from `report.winddown_outcomes`.
- Add `--winddown-participation FLOAT` (default 0.10) option, threaded to
  `run_rebalance`.

## Fail-closed invariants (must all hold)

1. **Never open.** Force `target={}` and call `reconcile` directly; never call
   an orphan strategy's `target_positions()` (a quarantined strategy could emit
   BUY signals and re-open). reconcile-with-empty-target is structurally
   flatten-only.
2. **Cap every exit** at the ADV participation fraction.
3. **Dry-run does not submit and does not zero snapshots** (else the next live
   reconciliation expects 0 while Alpaca still holds N → false halt). Dry-run
   MAY record orphan rows with `dry_run=True` for visibility.
4. **Atomic snapshot zero** — zero the orphan snapshot in the same block as
   submission (live only).
5. **Guard `slug not in REGISTRY`** — skip + log, never crash.
6. **No allocation to orphans** — wind-down is a separate, allocation-free path;
   `allocate_capital` already returns 0.0 for non-LIVE.
7. **Orphan set derived from governance state at runtime**, decoupled from any
   mutable flag.
8. **Emergency halt + market-open guards still gate everything** — the wind-down
   block sits inside `run_rebalance` after those guards, so a halt or closed
   market skips it too.

## Testing

`tests/live/test_winddown.py` (pure) + extensions to the rebalance/safety tests:
- `capped_qty`: caps when ADV binds; passes through when it doesn't; `adv<=0`,
  `price<=0`, non-finite → 0.
- `winddown_orders`: a long snapshot → SELL-only, never a BUY-to-open; a short
  snapshot → BUY-to-cover only; ADV cap shrinks qty; a symbol with no bars →
  dropped into `skipped` (not silently lost).
- `detect_orphans`: returns non-LIVE slugs with non-empty snapshots present in
  REGISTRY; excludes LIVE slugs, empty snapshots, and de-registered slugs.
- `check_reconciliation` with `winddown_slugs`: orphan with N shares in snapshot
  + Alpaca → PASS (counted as expected); orphan flattened (snapshot 0, Alpaca 0)
  → PASS; live-strategy mismatch still FAILS.
- **Integration (the key one):** two-rebalance convergence — rebalance 1 detects
  an orphan, emits capped exits, zeroes its snapshot; rebalance 2 sees it flat
  and reconciliation passes; the live strategy trades normally throughout.
- Dry-run: no submission, snapshot unchanged.

## How it could fail

- **Participation too low for a thin name** → multi-rebalance wind-down (by
  design; acceptable). For the current liquid book the cap never binds.
- **A de-registered orphan** (snapshot exists, slug gone from REGISTRY) → skipped
  with a logged warning; its position is *not* wound down automatically (needs a
  manual exit). Flagged, not silent. (None currently; all five quarantined slugs
  are registered.)
- **Bars stale/missing for an orphan universe** → ADV 0 → exits no-op that day,
  logged in `skipped`; retried next rebalance once bars refresh.

## Out of scope

- Reviving the quarantined strategies (separately disproven for trend; the
  others fail more gates).
- The `bootstrap_lower` gate-calibration audit (separate pending deliverable).
- Choosing a non-default participation policy (0.10 is a sensible, documented
  default; tunable via the CLI flag).
