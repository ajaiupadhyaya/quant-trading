# Market-impact model — design spec

**Date:** 2026-05-28
**Charter gap:** #2 (realistic execution), slice 2b of 3.
**Status:** approved, pending implementation plan.

## Motivation

`docs/CHARTER.md` principle 2 requires modeling **market impact** — "a strategy
that ignores these is not a result." The engine charges a flat per-fill
half-spread (`slippage_bps`) that does not scale with trade size, so a $1k order
and a $100M order in the same name cost the same in bps. This slice adds a
size-dependent impact cost and is the prerequisite for capacity (slice 2c).

Slice 2a (borrow/financing) already shipped. Capacity (2c) is the next slice and
is out of scope here.

## Model

**Square-root law of market impact** (Almgren et al. — the empirically dominant
form), added on top of the existing flat half-spread:

```
participation = trade_notional / trailing_dollar_adv
impact_bps     = impact_coef_bps * sqrt(participation)
```

- `trade_notional = |qty| * mid_price` at the fill.
- `trailing_dollar_adv` = mean of `close * volume` over the `adv_window` bars
  **strictly before** the fill bar (the fill bar's own volume is excluded → PIT,
  no lookahead).
- `impact_coef_bps` is the impact in bps at 100%-of-ADV participation
  (`participation = 1`). Default **100.0** — the right order of magnitude for the
  square-root law, but a **provisional** calibration placeholder (true impact
  needs execution data). Exposed in config.
- The impact is **additive** on the flat half-spread: the fill price moves by
  `(slippage_bps + impact_bps) / 1e4`. The flat `slippage_bps` keeps its meaning
  as the fixed spread/fixed cost; impact is the size-dependent term.

**Default-on.** Like slice 2a, this is a model change that should make results
more realistic. At current book sizes the strategies trade tiny fractions of
liquid-name ADV, so realized impact is near-zero today; it matters chiefly for
capacity (2c) at large AUM. Existing governance/validation evidence is refreshed
by re-running `quant validate` — expected, not a regression.

### Edge cases

Mirror the engine's tolerance for sparse data; return `0.0` impact (cannot
estimate) rather than raising, when:

- `trailing_dollar_adv <= 0`, non-finite, or there is no prior history
  (insufficient bars before the fill).
- `trade_notional <= 0` (e.g., zero-qty fill).

No impact cap in this slice: participation is naturally small for the liquid
universe, and capacity (2c) wants the uncapped square-root curve. A near-zero
ADV on a bad-data name could in principle produce a large impact — noted under
"How it could fail"; the `adv > 0` guard prevents a divide-by-zero.

## Architecture

### New unit — `quant/backtest/impact.py`

Pure, isolated, mirroring `financing.py` / `activity.py`. Takes plain values
(not `BacktestConfig`) so it stays standalone and avoids a circular import with
`engine.py`.

```python
def market_impact_bps(
    trade_notional: float,
    adv_dollar: float,
    impact_coef_bps: float,
) -> float:
    """coef * sqrt(trade_notional / adv_dollar), in bps. 0.0 when undefined."""


def trailing_dollar_adv(
    bars: pd.DataFrame,
    symbol: str,
    fill_ts: pd.Timestamp,
    window: int,
) -> float:
    """Mean close*volume over the `window` bars strictly before `fill_ts`.

    PIT: the fill bar's own volume is excluded. Returns 0.0 if there is no
    prior history or the (symbol, field) columns are absent.
    """
```

`bars` is the wide MultiIndex `(symbol, field)` frame the engine already holds;
`trailing_dollar_adv` reads `bars[(symbol, "close")]` and `bars[(symbol,
"volume")]` for rows with index `< fill_ts`, takes the last `window`, and
returns `mean(close * volume)`. Missing columns or empty slice → `0.0`.

### Config — `BacktestConfig` (engine.py)

Add two fields (defaults active because impact is on by default):

```python
    impact_coef_bps: float = 100.0   # square-root impact at 100% ADV participation; provisional
    adv_window: int = 21             # trailing bars for dollar-ADV (≈1 month)
```

### Engine wiring — `apply_costs` + `_execute_fill` (engine.py)

- `apply_costs` gains a parameter `impact_bps: float = 0.0`. The slipped fill
  price uses the **sum**: `slip = (config.slippage_bps + impact_bps) / 1e4`;
  `fill_price = mid * (1 ± slip)`. `slippage_cost = |qty| * |fill_price - mid|`
  is unchanged in formula and so now captures **spread + impact** together.
  Commission is unchanged.
- `_execute_fill` (which has `bars` and `ts` in closure scope) computes, before
  calling `apply_costs`:
  - `adv = trailing_dollar_adv(bars, sym, ts, config.adv_window)`
  - `notional = abs(qty) * mid`
  - `imp = market_impact_bps(notional, adv, config.impact_coef_bps)`
  - then `apply_costs(qty=qty, mid_price=mid, side=side, config=config, impact_bps=imp)`.

**Reporting:** impact is folded into the fill price, so it is captured in
`slippage_cost`, equity, returns, and every derived metric automatically. No new
trades-ledger column is added in this slice (avoids touching the trades schema,
tear-sheet, and reconciliation); a separate `impact_cost` line is a possible
follow-on. The `slippage_cost` column now means "spread + impact" — documented
in `apply_costs`.

### Cost-sensitivity sweep — unchanged

`_cost_sensitivity` (validation.py) does `dc_replace(base_config,
slippage_bps=bps)`, which overrides only the flat half-spread and leaves
`impact_coef_bps` / `adv_window` at their base values. So impact stays constant
across the 0/5/15/30bps sweep — the sweep keeps measuring **spread** sensitivity.
A combined spread+impact sweep is a future enhancement; impact sensitivity is the
domain of capacity (2c), which sweeps AUM.

## Testing

**`tests/backtest/test_impact.py` (pure):**
- `market_impact_bps`: `participation = 1` → exactly `impact_coef_bps`;
  `participation = 0.25` → `impact_coef_bps * 0.5`; concavity — `impact(2·notional)
  < 2 · impact(notional)`; `adv_dollar <= 0` → 0; `trade_notional <= 0` → 0;
  non-finite inputs → 0.
- `trailing_dollar_adv`: a known bars frame → exact `mean(close*volume)` over the
  strictly-prior window; the fill bar's own (spiked) volume is **excluded** (PIT);
  no prior history → 0; missing columns → 0; window longer than history → mean of
  what's available.

**`tests/backtest/test_engine_impact.py` (integration):**
- A large trade in a low-ADV name incurs more cost than the same trade in a
  high-ADV name (impact scales with participation).
- `impact_coef_bps = 0.0` → equity curve byte-for-byte identical to the
  pre-feature behavior (impact fully disabled).
- PIT / no-lookahead: spiking the **fill bar's own volume** does not change the
  impact charged (ADV uses only strictly-prior bars).
- `apply_costs` unit: with `impact_bps > 0`, the fill price moves by
  `(slippage_bps + impact_bps)/1e4` and `slippage_cost` reflects the combined
  move.

## How it could fail

- **Coefficient calibration:** `impact_coef_bps = 100` is a provisional
  order-of-magnitude placeholder; without execution data the absolute impact
  numbers should not be over-trusted. It is exposed in config and flagged in the
  docstring.
- **Bad-data ADV:** a name with a spuriously tiny `trailing_dollar_adv` (data
  glitch) would inflate participation and impact. The `adv > 0` guard prevents a
  divide-by-zero, but there is no cap — a future refinement if it bites.
- **Early bars:** trades in the first `adv_window` bars (no full trailing window)
  use whatever prior history exists, or `0` impact if none — slightly understates
  impact at the very start of a backtest.
- **`slippage_cost` conflation:** the column now blends spread and impact; a
  reader expecting pure spread will misread it. Documented in `apply_costs`; a
  separate column is deferred.

## Out of scope (this slice)

- Capacity (slice 2c) — the model here is its input.
- Per-name volatility scaling of impact (`coef · σ · √participation`).
- An impact cap; a separate `impact_cost` ledger column.
- Sweeping `impact_coef_bps` in the cost-sensitivity validation sweep.
- A permanent-vs-temporary impact split (single combined impact term only).
