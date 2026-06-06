# Risk-based capital allocation mode

**Date:** 2026-06-05
**Status:** Design — building directly on `main` (single-main workflow).
**Charter tie-in:** §3 portfolio construction / risk management. Closes the audit
gap "cross-strategy capital is equal-split, not risk-based (no Kelly, no vol weighting)."

## Problem

`quant/governance/allocation.py::allocate_capital` decides how live capital is
split across governance-LIVE strategies. Its default mode is `equal-live`
(1/N) — and that is what the live book runs. The other modes (`dsr-weighted`,
`capped-evidence-score`) weight by *evidence scores*, not by each strategy's
*risk*. So a low-vol defensive sleeve and a high-vol momentum sleeve get the same
capital, and the portfolio's risk is whatever falls out — not deliberately
balanced.

We want a **risk-based** allocation: weight inversely to each strategy's
volatility (risk parity) or by its fractional-Kelly fraction (edge / variance).

## Risk source: walk-forward OOS curves, not live history

The system went live recently, so per-strategy *live* return history is too short
to estimate vol/edge reliably. Each strategy already has a committed
**`data/backtests/<slug>/walkforward.parquet`** — a multi-year out-of-sample
equity curve (DatetimeIndex + `equity` column), referenced by
`ValidationEvidence.walkforward_path`. Daily returns = `equity.pct_change()`. This
is the robust, PIT-honest risk estimate (it's the same OOS evidence governance
already trusts to authorize the strategy). No live-P&L-attribution plumbing
needed.

## Design

### Config (`AllocationConfig`, extended)

| field | default | meaning |
|---|---|---|
| `mode` | `"equal-live"` | **unchanged default ⇒ zero live behavior change** |
| `max_weight` | `0.40` | (existing) per-strategy cap |
| `min_weight` | `0.05` | (existing) per-strategy floor |
| `kelly_fraction` | `0.5` | fractional-Kelly multiplier (half-Kelly) |
| `kelly_cap` | `1.0` | per-strategy Kelly clamp |
| `min_observations` | `60` | min daily returns required to trust a strategy's σ/μ |

New `AllocationMode` values: `"risk-parity"`, `"fractional-kelly"`.

### Pure core

```
strategy_risk(returns) -> (mean, std)            # daily μ, σ (ddof=1); NaN if < min_obs or degenerate
risk_based_raw_weights(returns_by_slug, live_slugs, mode, config) -> dict[str,float] | None
```

- **risk-parity:** `raw_i = 1 / σ_i`.
- **fractional-kelly:** `raw_i = fractional_kelly(μ_i, σ_i², kelly_fraction, kelly_cap)`
  (reuses `quant/sizing/components.fractional_kelly` — clamp(f·μ/σ², 0, cap)).
- Returns **`None`** (⇒ caller falls back to `equal-live`) if ANY live strategy
  lacks a usable estimate (missing curve, < `min_observations`, σ ≤ 0 / non-finite),
  or if every raw weight is ≤ 0. Fail-open and all-or-nothing: never silently
  weight a subset, never act on a strategy whose risk we can't measure.

`allocate_capital` gains an optional `returns_by_slug` param. For the two risk
modes it computes raw weights via the pure core (falling back to equal-live raw
when `None`), then runs the **existing** `_normalize_with_cap_and_floor` — so the
0.40 cap / 0.05 floor invariants are preserved identically.

### Impure edge (loader)

```
load_strategy_returns(evidence_by_slug, data_dir) -> dict[str, np.ndarray]
```

Reads each `evidence.walkforward_path` parquet → `equity.pct_change().dropna()`.
Best-effort: a slug whose parquet is missing/unreadable is simply absent from the
dict (⇒ triggers the fail-open fallback above). No exceptions escape.

### Wiring (`run_rebalance`)

`run_rebalance` gains `alloc_config: AllocationConfig | None = None`
(default `None` ⇒ `AllocationConfig()` ⇒ `equal-live` ⇒ **byte-for-byte today**).
When the configured mode is risk-based it loads returns and passes them.

**Observed-first comparison artifact:** regardless of the active mode,
`run_rebalance` computes BOTH the equal-live weights and the risk-based weights
(when curves are loadable) and writes
`data/governance/allocation_compare.<asof>.json` (active mode, equal-live weights,
risk-parity weights, fractional-kelly weights, per-strategy μ/σ, fallback reason
if any). This lets the user *see* what a risk-based split would do before
enabling it — the same shadow pattern used for Guard-5 and the sizing overlay.
The artifact is best-effort (never blocks a rebalance).

## Why this is safe and correct

- **Zero live change by default:** default mode stays `equal-live`; the new code
  paths only run when a non-default mode is configured.
- **Robust risk estimate:** uses years of committed OOS returns, not noisy live data.
- **Fail-open & all-or-nothing:** any unmeasurable strategy ⇒ full fallback to
  equal-live; never a half-risk-weighted book.
- **Invariants preserved:** weights still pass through the existing cap/floor
  normalizer (sum to 1, ≤ max_weight, ≥ min_weight).
- **Reuses existing math:** `fractional_kelly` from `quant/sizing/components`.

## Out of scope (deferred)

- Actuating a risk mode live (requires a conscious human enable — like WARN→BLOCK).
- Per-strategy *live* return attribution (separate plumbing; not needed here).
- Covariance-aware allocation (HRP across *strategies*, correlation terms) — this
  ships diagonal risk (inverse-vol / Kelly) first.

## Test plan (TDD)

1. `equal-live` default ⇒ identical to current `allocate_capital` (regression-pinned).
2. `strategy_risk`: correct μ/σ; NaN when < `min_observations` or σ = 0.
3. risk-parity: lower-vol strategy gets higher weight; two equal-vol ⇒ equal (pre-cap).
4. fractional-kelly: higher μ/σ² ⇒ higher weight; μ ≤ 0 ⇒ 0 contribution.
5. fallback: a missing/short curve for any live slug ⇒ exactly the equal-live result.
6. cap/floor invariants hold for risk modes (≤ max_weight, ≥ min_weight, Σ=1).
7. single live strategy ⇒ {slug: 1.0} for every mode.
8. loader: reads a real parquet to returns; missing path ⇒ slug absent (no raise).
9. integration: `run_rebalance` with default config writes no behavior change;
   with a risk mode, the compare artifact exists and allocation reflects the mode.

## Acceptance

- `quant/governance/allocation.py` extended + `load_strategy_returns`;
  `tests/governance/test_allocation.py` extended (+ new cases).
- `ruff` + `mypy --strict` clean; full suite green.
- Committed to `main`.
