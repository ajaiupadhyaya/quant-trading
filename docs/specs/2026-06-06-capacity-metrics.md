# Strategy capacity metrics (charter gap #2, slice 2c)

**Date:** 2026-06-06
**Status:** shipped — capacity column live on `quant combined-book`; e.g. high-turnover pairs ~$13M (impact-bound) vs low-turnover risk-parity ~$270M (participation-bound). Closes principle-3 metric set + gap #2.
**Charter tie:** CHARTER.md principle 3 (robust validation — *report Sharpe + maxDD +
turnover + **capacity***) and gap #2 slice 2c. Unblocked by the square-root
market-impact model (`backtest/impact.py`, slice 2b).

## Motivation

Turnover (slice 2a/activity.py) answers "how much does this strategy trade?".
**Capacity** answers the dual question the charter explicitly names: "how much
*capital* can this strategy run before its own footprint erodes the edge?" Without
it a backtest's Sharpe is a half-truth — a great signal that only works at \$1M is
not the same asset as one that works at \$1B.

We already compute, per fill, a PIT trailing dollar-ADV (`impact.trailing_dollar_adv`)
and a square-root impact cost. Capacity falls straight out of those, so this slice
is mostly *surfacing* existing quantities, not new modeling.

## Design

### Ledger change (one column)

`engine.py` already computes `adv = trailing_dollar_adv(...)` per fill but discards
it after charging impact. We now persist it as an `adv_dollar` column on the trade
ledger, so capacity is a **pure function of the ledger** — symmetric with how
`annualized_turnover` reads `qty`/`fill_price`. Backtest ledger tests use subset
column checks, so the addition is non-breaking; legacy ledgers without the column
score zero fills and report `binding="none"` rather than raising.

### `capacity_report(trades, equity_curve, *, max_participation, impact_coef_bps, impact_budget_bps)`

Two model-free AUM ceilings, both assuming capital scales fill notionals linearly
(the standard capacity assumption):

- **Participation ceiling.** Each fill trades `notional/adv` of a day's volume.
  Scaling AUM by `k` scales participation by `k`. The largest `k` keeping the
  **95th-percentile** fill at/under `max_participation` (default 10%) gives
  `participation_capacity = mean_equity * max_participation / p95`. p95 (not max) is
  the headline so a single illiquid fill can't dictate the number; `max_participation`
  (the field) still exposes the worst case.
- **Impact ceiling.** Square-root impact cost per fill is
  `notional * impact_coef_bps*sqrt(notional/adv)/1e4`; summed + annualized it is a
  fractional drag `g0` on equity. Because notional ∝ k and impact_bps ∝ sqrt(k), the
  drag grows as `sqrt(k)`, so the AUM at which it reaches `impact_budget_bps`
  (default 100 bps/yr) is `impact_capacity = mean_equity * (budget/g0)^2`. `inf`
  when impact is off (never binds).

`capacity_aum = min(participation, impact)`; `binding` names the constraint
("participation" | "impact" | "none").

### Surfaces

- `quant combined-book` CLI table gains a **Capacity** column (`$X.YM (part|impact)`).
- HTML tear-sheets (single + combined) gain a **Capacity (AUM ceiling)** row beside
  Turnover, via two new defaulted `_MetricsBundle` fields (`capacity_aum`,
  `capacity_binding`) — backward-compatible.

## Honesty / charter alignment

- Research/reporting only — reads the ledger, drives no sizing, order, or live path.
- PIT throughout (ADV is the engine's strictly-prior trailing window).
- Conservative-by-default knobs (10% participation cap, 100 bps impact budget) are
  caller-overridable; p95 headline avoids both outlier-domination and silent
  truncation (max is still reported).
- Closes the last open charter gap (2c), completing principle 3's metric set.

## Tests

`tests/backtest/test_activity.py` (capacity block): empty/legacy-ledger → none;
participation hand-calc; room-to-grow below cap; impact hand-calc + binding
selection; impact-binds-before-participation; p95 robustness to a lone illiquid
outlier; scale-homogeneity. Full backtest suite stays green with the ledger column.
