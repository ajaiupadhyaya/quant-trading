# ARIMA conditional-mean modeling (charter techniques — honest benchmark)

**Date:** 2026-06-06
**Status:** shipped — completes the charter's ARIMA/GARCH technique set. SPY result is the documented EMH negative (see "Result" below).

## Result + a gate-rigor finding

On SPY daily returns (3967 OOS steps, best ARIMA(2,0,0)): conditional IC +0.024
(t=1.50, **not** significant), hit-rate 51.0%, **MSE 1.009 — worse than the
unconditional baseline**. `passes = False`. The documented EMH outcome.

A first cut of the gate **wrongly passed** SPY (DSR 0.90 / PSR 0.94 on a *cost-free*
sign-strategy) — a real methodological trap worth recording: a daily-frequency
directional tilt with a faint, possibly-real short-term-reversal signature can clear
a Sharpe gate while (a) ignoring the transaction costs daily flipping incurs and
(b) not even beating the unconditional mean as a point forecast. Two charter-mandated
hardenings fixed it: a per-flip `cost_bps` (default 2.0) on the sign-strategy, and a
hard `mse_ratio < 1` requirement in `passes` (a *mean* model must out-forecast the
mean). After both, SPY reads "directional tilt clears DSR/PSR but the point forecast
does NOT beat the baseline — not a real conditional-mean edge." The gate remains a
true test: a genuine low-turnover AR(1) edge still passes all three bars
(mse_ratio 0.72, DSR/PSR 1.0).

**Charter tie:** Techniques row — "ARIMA/GARCH time-series modeling." GARCH (the
variance half) shipped. This is the **mean half**, completing the named set.

## Why build a model we expect to fail

Daily equity returns have essentially no linearly-predictable conditional mean —
that is the efficient-market prior. The honest way to *establish* that (rather than
assert it) is to fit the canonical conditional-mean model, evaluate it the same way
every other signal is evaluated (walk-forward, one-step-ahead, DSR/PSR-gated), and
let the negative result stand on the record. That negative result is the deliverable:
it is the empirical justification for the whole architecture — **the system forecasts
variance (GARCH) and the cross-section (factors), not the daily mean, *because* the
daily mean is unforecastable.** A documented, reproducible "no edge" is worth more
than a hand-wave.

The same machinery would also *detect* a real edge if one existed (verified on a
synthetic AR(1) with genuine structure), so this is a true test, not a rigged one.

## Design

New `quant/forecast/arima.py`, pure numpy, deterministic, dependency-free (no
`statsmodels`):

- **ARMA(p,q) by Hannan–Rissanen**: two deterministic OLS stages — (1) a long AR(m)
  fit to recover residual estimates, (2) regress the series on its own lags plus the
  lagged residuals to get `phi` (AR) and `theta` (MA). Optional differencing `d`
  (ARIMA) via `np.diff`, integrated back for the level forecast. An invertibility
  guard rescales `theta` if `sum|theta| >= 1` so the residual recursion can't diverge.
- `fit_arima(y, config) -> ARIMAModel | None` (None on too-few/degenerate data).
- `arima_forecast_next(model, y) -> float | None` — one-step-ahead, integrating `d`.

### Honest evaluation (same bar as every other signal)

- `walk_forward_arima_eval(y, ...)`: expanding-window, one-step-ahead. Collects
  `(forecast, realized)` pairs → OOS information coefficient (corr), its t-stat,
  directional hit-rate, and model MSE vs the **zero-forecast** (no-predictability)
  benchmark MSE.
- `arima_research_verdict(y, grid)`: builds a sign-following return series
  `sign(forecast)·realized_next` for the chosen (p,q), then **deflates its per-step
  Sharpe against the per-(p,q)-trial Sharpes of the whole grid** (the honest
  multiple-testing set) via `deflated_sharpe`, plus `probabilistic_sharpe`. Gates at
  the live bar (DSR ≥ 0.30, PSR ≥ 0.70). Reports eligibility; promotes nothing.
- CLI `quant forecast arima-eval` prints IC / t-stat / hit-rate / MSE-ratio / DSR /
  PSR / verdict with a research-only + EMH-documentation banner.

## Honesty / charter alignment

- Research-only: wired to no strategy, tilt, sizing, or order path.
- PIT (only past returns enter each fit), deterministic, dependency-free.
- Multiple-testing-corrected (DSR over the (p,q) grid), distrusts in-sample.
- Expected outcome on daily SPY returns: `passes = False` (no edge) — the documented,
  charter-anticipated result.

## Tests

- HR recovers a known AR(1)/ARMA(1,1) coefficient set within tolerance; forecast
  correlates with truth on a structured series.
- White-noise input → near-zero forecast, IC ≈ 0, verdict `passes = False`.
- Differencing `d=1` models a random-walk-in-levels' increments.
- Determinism (identical inputs → identical model); None on short/degenerate input.
- `walk_forward_arima_eval` runs, `n_oos > 0`, hit-rate in [0,1].
- Sanity: a strong synthetic AR(1) returns series yields IC > 0 and hit-rate > 0.5
  (the machinery *can* find a real edge), while real/white-noise daily returns do not.
