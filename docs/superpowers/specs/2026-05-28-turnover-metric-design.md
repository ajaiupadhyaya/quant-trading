# Turnover metric — design spec

**Date:** 2026-05-28
**Charter gap:** #1 (turnover + capacity), slice 1 of 2.
**Status:** approved, pending implementation plan.

## Motivation

`docs/CHARTER.md` principle 3 requires every strategy report **Sharpe, max
drawdown, turnover, and capacity** — not just returns. The codebase reports
Sharpe / Sortino / max-drawdown / win-rate / CAGR (`quant/backtest/metrics.py`)
but has **no turnover and no capacity metric**. This spec closes the turnover
half.

Capacity is **deliberately deferred** to charter gap #2: a real capacity number
("max AUM before market impact erodes the alpha") requires the market-impact
model that gap #2 builds. Computing it now against a flat-bps placeholder would
produce exactly the kind of unfounded "looks-like-a-result" number the charter
tells us to flag. Capacity (both a model-free proxy and the impact-adjusted
figure) lands with gap #2 so it is never shown half-built.

## Definition

**Fills-based, one-way, annualized turnover.** Computed from the actual trade
ledger (`BacktestResult.trades`), so it reflects real fills — including
zero-crossing flatten-and-reopen and slipped fill prices — rather than an
idealized weight diff.

Given a trade ledger and the daily equity curve over the same window:

```
traded_notional = Σ |qty| · fill_price          # over all rows in the ledger
one_way         = traded_notional / 2            # full round-trip = 100%
annualized      = (one_way / mean_equity) · (periods_per_year / n_days)
```

- `qty` is stored in the ledger as a positive magnitude with direction carried
  in `side`, so `Σ |qty|·fill_price` is the total two-way dollar volume traded.
- Halving yields the one-way convention: buying then later selling the same
  notional on a constant-equity book over a year reads as 100% turnover.
- `mean_equity = equity_curve.mean()`; `n_days = len(equity_curve)`.
- Annualization scales the whole-window one-way fraction to a per-year rate.

### Edge cases

Mirror the `metrics.py` contract — undefined results return `0.0`, never raise,
so tear-sheet rendering never breaks:

- empty trade ledger → `0.0`
- empty equity curve, or `mean_equity ≤ ε`, or `n_days == 0` → `0.0`

## Architecture

### New unit — `quant/backtest/activity.py`

A focused module for **trade-activity metrics**. Turnover now; the natural home
for capacity in gap #2.

```python
def annualized_turnover(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float: ...
```

It deliberately takes the trade ledger + equity curve, **not** the
`returns: pd.Series` that `metrics.py` is built around — which is exactly why it
is a separate module rather than an addition to `metrics.py`. The module
docstring states this so the boundary is self-explaining.

**Inputs / contract:**
- `trades`: a `BacktestResult.trades`-shaped frame — must expose `qty` and
  `fill_price` columns. An empty frame (no rows, or the canonical empty-columns
  frame the engine emits) is valid and yields `0.0`.
- `equity_curve`: a `BacktestResult.equity_curve`-shaped daily Series.
- Returns: a non-negative float (annualized one-way turnover fraction; `1.0`
  = 100% of the book turned over one-way per year).

### Integration — `quant/backtest/tearsheet.py`

`_MetricsBundle` already carries non-returns fields (`n_trades`,
`starting_equity`, `ending_equity`), so turnover fits its existing shape.

- Add `turnover: float` to `_MetricsBundle`.
- Compute it at all three build sites from their own `(trades, equity_curve)`
  pair:
  - `write_tearsheet` → `annualized_turnover(result.oos_trades, result.oos_equity_curve)`
  - `write_combined_tearsheet` → `annualized_turnover(result.trades, result.equity_curve)`
  - per-strategy rows → `annualized_turnover(sub.trades, sub.equity_curve)`
- `templates/tearsheet.html.j2`: add a "Turnover (ann.)" row to the headline
  metrics table and a "Turnover" column to the per-strategy combined table.

### Other scalar-metric surfaces

During planning, enumerate any registry-logged metrics dict and CLI summary
output (`quant backtest` / `quant validate` / combined-book reporting) that
already print Sharpe / MaxDD, and add turnover alongside — so the charter's
"report turnover" holds everywhere those metrics appear, not only in the HTML.
Turnover is a **reported** metric, not a validation gate.

## Testing — `tests/backtest/test_activity.py`

- **Hand-computed fixture:** a known small ledger + flat equity over 252 days →
  asserts the exact expected annualized value.
- **Convention check:** one full round-trip (buy N notional, later sell N) on a
  constant-equity book over 252 days → `1.0`.
- **Annualization:** identical trades over 126 days → 2× the 252-day figure.
- **Homogeneity invariant:** scaling notional and equity by the same factor
  leaves turnover unchanged.
- **Edge cases:** empty ledger → `0.0`; empty equity → `0.0`.
- **Render check:** one existing tear-sheet test updated to assert the turnover
  row appears in the output.

## Out of scope (YAGNI)

- Capacity (model-free proxy and impact-adjusted) — gap #2.
- Turnover-over-time chart / per-rebalance turnover series.
- Turnover as a validation/governance gate.

## How it could fail

- **Definitional drift:** if a reader expects two-way (gross) turnover, the
  one-way figure looks half as large as they assume. Mitigated by labeling the
  metric "one-way" in the spec and using "Turnover (ann.)" with the convention
  documented.
- **Mean-equity sensitivity:** dividing by `mean_equity` over a window with a
  large drawdown or strong compounding shifts the denominator vs a
  point-in-time normalization. Acceptable for a headline figure; per-period
  turnover (deferred) would be the finer view.
- **Sparse ledgers:** a strategy that trades a handful of times produces a
  noisy annualized number on short windows — inherent to annualizing, not a
  bug; the `n_trades` field already shown alongside gives the reader context.
