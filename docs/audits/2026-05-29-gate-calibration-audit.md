# Gate-Calibration Audit: is `bootstrap_lower` the right live-gate?

**Date:** 2026-05-29
**Scope:** the daily-batch system's governance gate. Independent of the intraday work.
**Status:** analysis complete. **The decision to change any gate is the user's** — this audit
provides the basis; it deliberately does not alter the gate (that would be the user's call, and
changing a gate to admit a preferred strategy would violate the charter).

## Question

Strategy governance promotes a strategy to `live` only if it passes five gates: Deflated Sharpe,
Probabilistic Sharpe, **`bootstrap_lower`**, regime, and holdout. The `bootstrap_lower` gate is

```python
# quant/backtest/validation.py:227
gate_boot = ci is not None and ci.total_return_p05 > 0.0
```

i.e. the **5th percentile of the bootstrapped cumulative OOS total return must exceed zero**. Two
concerns motivated this audit:

1. That metric is **not risk-adjusted** — a higher-volatility strategy has a fatter left tail of
   cumulative return and so a more-negative `total_return_p05`, *even at the same Sharpe*. The gate
   may therefore penalize volatility rather than lack of edge.
2. `bootstrap_ci` already computes **`sharpe_p05`** (a risk-adjusted lower bound) at
   `quant/backtest/bootstrap.py:117`, but it is **never gated and never persisted** — a principled
   alternative is being calculated and discarded.

## Two data-integrity findings (surfaced before any analysis)

- **The live manifest was stale.** Every entry in `data/governance/validation_manifest.json` was
  dated 2026-05-26/27 — *before* financing + market-impact costs became default-on. The live roster
  was keyed off validation that never charged realistic costs.
- **The manifest and the walk-forward `.parquet` artifacts were mutually inconsistent** (different
  runs — e.g. trend's recorded DSR 0.52 was incompatible with the parquet's Sharpe 1.08), evidently
  from partial regenerations. Neither could be trusted.

Because of this, the audit **regenerated clean, internally-consistent, cost-on OOS returns** rather
than reusing the persisted artifacts.

## Method

Faithfully mirrored `quant validate` (the authoritative path): `BacktestConfig()` cost-on defaults
(slippage + commission + borrow + financing + market-impact), walk-forward train=5y / test=1y /
step=6mo, holdout=1y, window 2010-01-01 → 2026-05-26. Bootstrap = stationary-block, 1000 resamples,
block length 5, seed 0 (the gate's exact config). Computed both `total_return_p05` (the current
gate) and `sharpe_p05` (the candidate) from the **same** OOS series. Added a **seed×block stability
sweep**: 30 seeds × block ∈ {3,5,10,21} = 120 configurations, to see how often each candidate gate's
pass/fail flips with the RNG.

The four price/ETF strategies were re-run. `multi-factor` and `pairs` were excluded: they fail 4 of
5 gates (DSR ≈ 0.007 / 0.02) regardless of the bootstrap gate, and their EDGAR backfill is
prohibitively slow (~90 min each) — their fate does not hinge on this gate. *(That exclusion is
noted rather than hidden; including them would not change any conclusion.)*

All four strategies share the **identical 2015-01-02 → 2024-12-31 OOS window (2516 days)** under this
faithful run — so the window-length confound I initially suspected was an artifact of the stale,
inconsistent parquets, **not** a real feature. The live bias is therefore purely the volatility
penalty.

## Results (cost-on)

| Strategy | Ann. ret | Ann. vol | Sharpe | `total_return_p05` (current gate) | `sharpe_p05` (candidate) | Verdict |
|---|---|---|---|---|---|---|
| **defensive-etf** | 8.76% | 15.1% | 0.63 | **+0.012** ✅ | **+0.077** ✅ | passes both |
| trend | 4.33% | 9.2% | 0.51 | −0.052 ❌ | −0.018 ❌ | fails both |
| momentum | 4.58% | 11.4% | 0.45 | −0.150 ❌ | −0.101 ❌ | fails both |
| risk-parity | 0.27% | 8.6% | 0.07 | −0.372 ❌ | −0.499 ❌ | fails both |

Seed×block **stability** — fraction of 120 configurations in which each gate *passes*:

| Strategy | `total_return_p05 > 0` pass-rate | `sharpe_p05 > 0` pass-rate |
|---|---|---|
| **defensive-etf** | 90.8% | **100%** |
| trend | 10.8% | **59.2%** |
| momentum | 0% | 0% |
| risk-parity | 0% | 0% |

## Findings

1. **The volatility bias is real and material — for trend specifically.** Under the risk-adjusted
   metric, trend passes **59%** of configurations; under the current raw-cumulative-return gate it
   passes only **11%**. Same strategy, same data — the gap is the volatility penalty. So the current
   gate *is* harsher on trend than a principled risk-adjusted gate would be. The concern that
   motivated this audit is **confirmed**.

2. **But correcting the bias does not produce a confident new live candidate.** Even on `sharpe_p05`,
   trend is a **coin-flip** (59%; the point estimate at seed 0 still fails at −0.018). A disciplined
   operator does not promote a strategy that clears the bar in roughly half of RNG seeds.

3. **momentum and risk-parity fail under every metric and every configuration** (0% pass-rate
   either way). They have no robust edge after realistic costs; no gate change touches them.

4. **defensive-etf passes robustly under both** (`sharpe_p05` passes 100% of configs). It is
   correctly live. *Caveat:* its cost-on `total_return_p05` is **+1.2%**, far thinner than the stale
   manifest's +5.9% — realistic costs ate most of the margin. Still a pass, but worth monitoring.

## Recommendation (the decision is the user's)

- **Adopt a risk-adjusted lower-bound gate** (`sharpe_p05 > 0`) in place of, or alongside,
  `total_return_p05 > 0`. It is the methodologically correct choice: it isolates risk-adjusted edge
  from volatility, and it uses a metric the pipeline **already computes** (currently discarded).
  Persist `sharpe_p05` in the manifest.
- **Require a robustness margin, not a point estimate.** Given how much trend's verdict swings with
  the RNG seed, gate on something like "`sharpe_p05 > 0` across ≥ X% of a seed/block sweep" or
  "`sharpe_p05` above a small positive threshold," applied **uniformly to all strategies**.
- **Net effect on the live roster: none.** Under the improved gate, defensive-etf remains the sole
  robust pass; trend becomes a *borderline "watch"* (not a promote); momentum and risk-parity remain
  clear fails. **No new strategy should go live on the strength of this change.**

### Anti-gaming check (charter compliance)

This recommendation is **not** gate-gaming, by three tests:
1. **Uniform** — applied to all strategies, not carved out for a favorite.
2. **Pre-existing principled metric** — `sharpe_p05` was always computed; it is not reverse-engineered
   to pass trend.
3. **Does not change who goes live** — the roster is unchanged; trend still does not qualify.

If any future proposal fails one of these tests (e.g. "lower the threshold until trend passes"),
that *would* be gaming and should be rejected.

## Operational follow-ups (separate from the gate decision)

- The weekly `validation-governance` cron should regenerate `validation_report.json` +
  `walkforward.parquet` **atomically** so the manifest and artifacts can never drift out of sync
  again. The next weekly run will also refresh the manifest under cost-on defaults automatically.
- Track defensive-etf's thin cost-on margin (+1.2%); if it turns negative on a future run, governance
  will (correctly) quarantine it and the account would hold no live strategy — worth an alert.

## Reproducibility

Regeneration + sweep script and raw results retained for the session; the run is fully determined by
the config above (cost-on `BacktestConfig()` defaults, walk-forward 5/1/6mo, holdout 1y, bootstrap
1000×block-5×seed-0, sweep 30 seeds × blocks {3,5,10,21}). To reproduce in-tree, run
`quant validate <slug> --start 2010-01-01 --end 2026-05-26` per strategy (cost-on by default) and
bootstrap the resulting OOS returns.
