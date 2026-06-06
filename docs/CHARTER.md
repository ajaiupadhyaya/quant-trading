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
| 2 — Realistic execution | ✅ Met | `backtest/engine.py` models flat `slippage_bps` + `commission_bps` + a 0/5/15/30bps cost-sensitivity sweep; daily **short-borrow + margin-financing** costs accrue (actual/365, `backtest/financing.py`); **size-scaled square-root market impact** (ADV-based, `backtest/impact.py`) is charged per fill; per-fill PIT dollar-ADV is persisted to the ledger and **capacity** (participation + impact-adjusted AUM ceilings) is reported (`activity.py`). All on by default. |
| 3 — Robust validation | ✅ Met | Walk-forward + CPCV + DSR + PSR + bootstrap + regime stress + OOS holdout; `metrics.py` reports Sharpe/Sortino/maxDD/win-rate/CAGR; `activity.py` reports **annualized turnover** AND **capacity** (model-free participation ceiling + impact-adjusted ceiling, binding AUM) on tear-sheets + `quant combined-book`. |
| 4 — Overfitting guard | ✅ Met | Deflated Sharpe (`dsr.py`, Bailey–López de Prado multiple-testing correction), CPCV, walk-forward param grids. |
| 5 — Reproducibility | ✅ Met | Run registry logs every backtest (params + kind), deterministic governance manifests, git history as audit trail. (RNG seeding audit pending.) |
| Techniques | ✅ Met | Factor models, momentum, mean-reversion, stat-arb, HAR-RV vol, **GARCH/GJR-GARCH vol** (`forecast/garch.py`, GJR promoted advisory-primary), **DSR-gated gradient-boosting** (`forecast/gbm.py`), and **ARIMA conditional-mean** (`forecast/arima.py`, Hannan-Rissanen, DSR/PSR + cost + beat-baseline gated). ARIMA's documented OOS result on SPY is the EMH negative (no conditional-mean edge — `passes=False`), which is *why* the system forecasts variance + cross-section, not the daily mean. |

### Open gaps being closed

1. **Turnover + capacity metrics** (principle 3) — turnover ✅ shipped; capacity ✅ shipped (slice 2c, `activity.py` `capacity_report`). **Gap fully closed.**
2. **Borrow + market-impact costs** (principle 2) — borrow/financing ✅ (slice 2a, `backtest/financing.py`); square-root market impact ✅ (slice 2b, `backtest/impact.py`); capacity ✅ (slice 2c, `activity.py` — model-free participation ceiling + impact-adjusted ceiling, reading the per-fill `adv_dollar` now persisted on the ledger). All on by default. **Gap fully closed.**
3. **ARIMA/GARCH volatility modeling** (techniques) — GARCH(1,1) + GJR-GARCH ✅ shipped (`forecast/garch.py`, hand-rolled QMLE under variance targeting), slotted into the existing one-day-ahead walk-forward race (`forecast/vol.py`, opt-in `include_garch`). Validated OOS on SPY (3967 days): GJR-GARCH wins the QLIKE race and GARCH beats the HAR incumbent (Diebold-Mariano p=0.045). Advisory/shadow — drives no sizing until a conscious promotion. ARIMA conditional-mean deferred (see Techniques row).
4. **Gradient-boosting signal layer** (techniques) — ✅ shipped (`forecast/gbm.py`, hand-rolled deterministic GBM; strict DSR≥0.30 / PSR≥0.70 gating in `forecast/factor.py`). Research-only; promotes nothing automatically.

Each closes through the repo's standard spec → plan → build discipline under
`docs/superpowers/`.
