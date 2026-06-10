# Strategy rehabilitation — honest re-validation under evidence schema v2

**Date:** 2026-06-10
**Author:** rehab effort (Phase 9 alpha)
**Status:** in progress

## Context

The 5 quarantined strategies (momentum, multi-factor, pairs, risk-parity, trend)
were last validated under **evidence schema v1** (~2026-05-26/27). The gate
methodology changed to **v2** on 2026-06-02: the Deflated Sharpe Ratio (DSR) now
deflates against the *full walk-forward grid-trial set* (windows × grid combos),
not the CPCV resample paths. v2 is strictly harsher for strategies with large
parameter grids, because DSR's deflation grows with both the trial count `N` and
the dispersion of trial Sharpes.

**The on-file quarantine reasons are therefore stale.** Step 1 of any honest
rehab is to re-baseline under v2. This doc pre-registers the hypotheses and
committed changes *before* re-validation, so results cannot be read as
gate-chasing.

## Honest guardrails (binding)

1. **Pre-register every change here before re-validating.** No iterating the grid
   until a gate flips green.
2. **Never lower a threshold.** Honor the DSR one-way trap. A documented null is a
   valid, successful outcome.
3. **Reject degenerate gaming.** Pure vol-shrink does NOT honestly fix the
   bootstrap p05 gate (scaling a negative p05 toward zero never flips its sign).
4. **Grid reduction is legitimate only on genuine a-priori economic grounds** that
   would convince a skeptical PM independent of the gate — and must be committed,
   not re-chased.

## v2 re-baselines (measured 2026-06-10)

| Strategy | DSR (v1→v2) | PSR | bootstrap p05 | regime | holdout | v2 failing gates |
|----------|-------------|-----|---------------|--------|---------|------------------|
| trend    | 0.52 → **0.138** | 0.965 | **−1.5%** | 4/4 ✅ | +14.6% ✅ | DSR, bootstrap |
| momentum | (pending v2) | | | | | (pending) |

Trend's DSR collapse (0.52→0.14) is entirely the v2 trial-set deflation: 19
windows × 12 grid combos = 228 trials, E[max of 228 normals] ≈ 2.85σ. The signal
is real (PSR 0.965); the *declared search* is what's being penalized.

## Trend — root cause (OOS 2015–2024)

CAGR 4.91%, vol 9.59%, Sharpe 0.55, max DD −17.08% (2018-10-29). The negative
bootstrap tail is dominated by **trend whipsaw in sharp-reversal regimes**:
2018 (−14.6%, worst quarter −8% ending 2018-07, worst day −5.8% on 2018-07-02),
with secondary bleeds in 2015 (−6%), 2016 (−4%), 2023 (−7%). The drift is strong
(2017 +26%, 2020 +20%, 2021 +21%) but the reversal tail is heavy enough that
block-resampling it yields a slightly-negative cumulative path 5% of the time.

Two **independent** levers:
- **DSR** ← caused by the 228-trial grid (not weak signal).
- **bootstrap p05** ← caused by reversal whipsaw (needs a genuine tail-shape fix).

## Trend — Change 1 (PRE-REGISTERED, this iteration)

**Tighten the declared hypothesis space to the strategy's design defaults.**

Old grid (12 combos): `vol_target_annual ∈ {0.08,0.10,0.12}` ×
`allow_short ∈ {True,False}` × `lookbacks_months ∈ {(3,6,12),(1,3,6,12)}`.

New grid (2 combos): `lookbacks_months ∈ {(3,6,12),(1,3,6,12)}` only.
Defaults committed: `allow_short=False`, `vol_target_annual=0.10`.

**Economic priors (hold independent of the gate):**
- *Long-only* — broad-ETF trend is conventionally long/flat; shorting beta on
  secularly-drifting equity/bond/commodity ETFs is a structural drag and adds
  high-variance trials. The live blessed strategy (defensive-etf-allocation) is
  long-only — this is the house style.
- *Fixed 10% vol target* — vol targeting commits to a single risk budget; the
  target is a risk preference, not alpha, and OOS Sharpe is ≈invariant to it.
  Searching over it is a category error that only inflates the trial count.

The lookback ensemble (2 options) remains a genuine, declared search.

**Predictions (pre-committed):**
- Trial set 228 → ~38 (19 windows × 2). DSR should rise materially (less
  multiple-testing penalty + removal of high-variance short trials).
- Bootstrap p05 may improve slightly (removing the allow_short=True windows of
  2019/2022–24) but is **not expected to pass** on this change alone — the 2018
  whipsaw is in a long-only window already. If DSR passes but bootstrap still
  fails, Change 2 will target the whipsaw tail (a genuine risk-management change,
  pre-registered separately).
- PSR, regime, holdout expected to remain passing.

If this change does not honestly clear the gates it is meant to, trend stays
quarantined and the null is recorded here.

### Trend — Change 1 RESULT (2026-06-10) — ✅ PASSES ALL FIVE GATES

| Gate | v2 baseline | after Change 1 | threshold |
|------|-------------|----------------|-----------|
| Deflated Sharpe | 0.138 | **0.502** ✅ | ≥0.30 |
| Probabilistic Sharpe | 0.965 | 0.990 ✅ | ≥0.70 |
| Bootstrap p05 | −1.5% | **+9.1%** ✅ (1000 resamples) | >0 |
| Regime | 4/4 ✅ | 4/4 ✅ | ≥50% |
| Holdout | +14.6% ✅ | +14.8% ✅ | >0 |

Walk-forward selects `lookbacks_months=(3,6,12)` (drops the whippy 1-month
component). New OOS curve (2015–2024): CAGR 6.65% (was 4.91%), vol 10.13%,
**Sharpe 0.69 (was 0.55)**, max DD −17.4% (≈unchanged — 2018 still −14.8%).

**Skeptical verification (per the "re-run at production fidelity" lesson):**
- Bootstrap p05 is NOT a seed artifact: at **5000 resamples it is +13.4% across
  seeds {0,1,2,7,42}** (median total return +88.8%). Stable and far from zero.
- The tail fix did NOT come from cutting the worst drawdown (max DD ≈unchanged).
  It came from a genuine Sharpe/drift lift — the good years improved (2019
  +9.6%→+18.4%, 2020 +19.9%→+25.8%, 2023 −6.9%→−3.5%), raising the whole
  resampled distribution above zero.
- **Honest caveat:** the bootstrap pass is a *fortunate emergent consequence* of
  the cleaner grid selecting the smoother `(3,6,12)` ensemble — not a designed
  tail fix. The lookback choice remains a genuine, DSR-counted search (2 options),
  so the deflation is honest. Change 2 (a designed tail fix) was NOT needed.

**Verdict:** trend earns LIVE under v2 honestly. Blessing (`governance refresh`)
and any live-allocation change (`run_rebalance`) are explicit manual steps —
pending operator greenlight.

### Trend — DEPLOYED LIVE (paper) 2026-06-10

- Canonical evidence at house standard (5000 bootstrap resamples): DSR 0.5022,
  PSR 0.9904, bootstrap p05 **+0.1333**, holdout +0.1477 — all gates pass.
- `governance refresh` → trend `quarantined → live`. Allocation (0.4 per-strategy
  cap): defensive-etf 0.4 / trend 0.4 / 0.2 cash.
- `quant rebalance --derisk-actuate` (engine flagged crisis/elevated → de-risk
  ×0.75 applied, honoring the one-way risk-off signal): 5 netted account orders,
  all **submitted = filled**, 0 rejected/partial/missing (recon 2026-06-10).
  trend holds SPY/DBC/VNQ/EFA/EEM; per-strategy ledger sums exactly to account
  targets (shared SPY/DBC/EEM netted across the two strategies).
- Code change: `quant/strategies/trend_following.py` param_grid tightened to the
  lookback ensemble only (uncommitted working-tree change; daemons run the
  working tree). Full test suite re-run after the change.

## Momentum — v2 re-baseline (2026-06-10)

| Gate | v1 | v2 | pass |
|------|----|----|------|
| DSR  | 0.64 | **0.119** | ❌ |
| PSR  | 0.93 | 0.928 | ✅ |
| bootstrap p05 | −12.8% | **−14.8%** | ❌ |
| regime | 50% | 2/4 ✅ | ✅ |
| holdout | +15.7% | +11.8% ✅ | ✅ |

DSR collapsed 0.64→0.119 — the 81-combo grid (3×3×3×3) × 19 windows ≈ 1539 trials,
E[max] ≈ 3.3σ, a self-inflicted deflation. Grid tightening is the obvious DSR
lever (same as trend). But the bootstrap gap (−14.8%) is far larger than trend's
(−1.5%) — momentum will likely need BOTH grid tightening AND a genuine tail fix.
Pre-registration of the momentum change is pending (next iteration).
