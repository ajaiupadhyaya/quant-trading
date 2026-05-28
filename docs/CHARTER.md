# Charter

This project's governing methodology. Every strategy, backtest, and feature in
this repo is held to it. When the charter and a quick result disagree, the
charter wins.

## Mission

Build and maintain a quantitative trading research system grounded in rigorous,
reproducible methodology. Apply established quant techniques: factor models,
mean-reversion and momentum signals, statistical arbitrage, time-series
modeling (ARIMA/GARCH), and ML where it's justified (gradient boosting, not deep
nets unless the data supports it).

## Core principles (priority order)

1. **No lookahead bias.** Every backtest must use point-in-time data; features
   may only use information available at decision time.
2. **Realistic execution.** Model transaction costs, slippage, market impact,
   and borrow/financing costs. A strategy that ignores these is not a result.
3. **Robust validation.** Use walk-forward / out-of-sample testing. Report
   Sharpe, max drawdown, turnover, and capacity — not just returns. Treat
   in-sample outperformance with suspicion.
4. **Guard against overfitting.** Penalize parameter-heavy strategies, account
   for multiple-testing when scanning signals, and prefer simple models that
   generalize over complex ones that fit noise.
5. **Reproducibility.** Seed randomness, version data and config, log every
   backtest run with its parameters and results.

## When proposing a strategy

State its **economic rationale**, its **assumptions**, and **how it could
fail**. Flag results that look too good before the operator does.

## Implementation status (2026-05-28)

How the codebase currently measures against each principle. Updated as gaps
close.

| Principle | Status | Where / gap |
|---|---|---|
| 1 — No lookahead | ✅ Met | PIT feature matrices, SEC EDGAR PIT fundamentals, Kalman/regime PIT, `atol=0` truncation-invariance property tests. |
| 2 — Realistic execution | ⚠️ Partial | `backtest/engine.py` models flat `slippage_bps` + `commission_bps` + a 0/5/15/30bps cost-sensitivity sweep. **Missing: borrow/financing cost on shorts (L/S books trade unmodeled shorts); size-scaled market impact (slippage is flat bps, not ADV/√-impact).** |
| 3 — Robust validation | ⚠️ Partial | Walk-forward + CPCV + DSR + PSR + bootstrap + regime stress + OOS holdout; `metrics.py` reports Sharpe/Sortino/maxDD/win-rate/CAGR. **Missing: turnover and capacity metrics, which this charter explicitly requires.** |
| 4 — Overfitting guard | ✅ Met | Deflated Sharpe (`dsr.py`, Bailey–López de Prado multiple-testing correction), CPCV, walk-forward param grids. |
| 5 — Reproducibility | ✅ Met | Run registry logs every backtest (params + kind), deterministic governance manifests, git history as audit trail. (RNG seeding audit pending.) |
| Techniques | ⚠️ Partial | Factor models, momentum, mean-reversion, stat-arb all present. **Missing: ARIMA/GARCH time-series modeling and gradient-boosting ML layer.** |

### Open gaps being closed

1. **Turnover + capacity metrics** (principle 3) — add to `metrics.py` + tear-sheet.
2. **Borrow + market-impact costs** (principle 2) — extend the engine cost model.
3. **ARIMA/GARCH volatility modeling** (techniques) — feeds vol-targeting / sizing.
4. **Gradient-boosting signal layer** (techniques) — strict OOS/DSR gating, given overfitting risk.

Each closes through the repo's standard spec → plan → build discipline under
`docs/superpowers/`.
