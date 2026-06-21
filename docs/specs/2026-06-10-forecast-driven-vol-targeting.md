# Forecast-driven vol-targeting — the "separate deliberate gate"

**Date:** 2026-06-10
**Status:** research / gate evaluation (pre-registered)

## Context

The vol forecast (`quant/forecast/vol.py`, GJR-GARCH primary → HAR → EWMA) is
OOS-validated for **accuracy** (GJR beat HAR, DM p=0.045; HAR beat EWMA, DM
p=0.012) but is **advisory-only** — it drives no sizing. Vol-targeting today
(`quant/sizing/`, and per-strategy) uses **trailing realized vol** (rolling
stdev). Every spec flags promotion-to-sizing as a **separate, deliberate gate**.

**The honest distinction:** accuracy ≠ sizing value. A forecast can win QLIKE yet
not improve the economic outcome of vol-targeting. This gate tests the economic
question directly, before anything drives sizing.

## Bridge (built, default-OFF)

- `forecast_vol_series(returns, model, refit_every, min_obs)` — PIT one-day-ahead
  annualised vol for each day, refit at `refit_every` cadence (no per-day refit),
  NaN warm-up; no-lookahead unit-tested.
- `SizingConfig.vol_source ∈ {"trailing"(default), "forecast"}` + model/refit knobs.
- `compute_gross(..., vol_override=)` — forecast feeds the vol-target component;
  `None` keeps the trailing path byte-identical (35 existing tests green).
- `compare_vol_source(returns, config)` → `VolSourceComparison` (return metrics +
  vol-tracking metrics for each source).

## Pre-registered gate (binding)

Run `compare_vol_source` (vol-target component ONLY, to isolate the vol-source)
on the LIVE strategies' OOS return curves (defensive-etf-allocation, trend) and
SPY, 2015–2024+. **Forecast-driven vol-targeting is promotable ONLY IF, on a
majority of the tested series:**

1. **Primary — tighter tracking:** lower `mad_from_target` (mean abs deviation of
   the rolling-63d realized vol from target) than trailing. Vol-targeting's actual
   job is to hold realized vol near target; this is the metric that matters.
2. **Do-no-harm:** Sharpe not materially worse AND max-drawdown not materially
   worse than the trailing variant.

If it fails (1) or harms (2), it is a **documented null** — trailing stays, the
forecast stays advisory. No threshold lowering, no cherry-picking the one series
that wins. Even on success, LIVE wiring is a further explicit human greenlight
(default stays shadow).

**Prediction:** the forecast should tighten tracking (it reacts faster to vol
shifts than a trailing window) — but on real strategy returns the effect may be
small, and the Sharpe/drawdown impact is genuinely uncertain (vol-timing
literature is mixed). Honest either way.

## RESULT (2026-06-10) — ✅ GATE PASSES (promotable to shadow)

A/B on the live strategies' OOS curves + SPY (2015–2024), vol-target component
isolated, target = each series' realized vol:

| Series | MAD-from-target (trail→fcast) | Sharpe | max-DD |
|--------|-------------------------------|--------|--------|
| defensive-etf | 0.0756 → **0.0584** ✅ | 1.34 → **1.45** | −22.1% → **−20.0%** |
| trend | 0.0557 → **0.0450** ✅ | 0.55 → **0.67** | −27.8% → **−21.2%** |
| SPY | 0.0371 → **0.0276** ✅ | 0.71 → 0.69 | −29.4% → −32.0% |

Gate: tighter-tracking **3/3**, Sharpe-do-no-harm 3/3, dd-do-no-harm 2/3 (SPY dd
slightly worse — the lone pure-index case).

**Robustness (skeptical re-runs, all tighter-tracking 3/3):** fixed target=0.12,
HAR model (not GJR), and exclude-2020 (no COVID spike) — **9/9 tighter-tracking**
across every variant. On the two LIVE strategies Sharpe improves in every variant;
SPY Sharpe is the only mixed case. Not a GJR-specific or COVID-driven artifact.

**VERDICT:** the forecast vol-source genuinely improves vol-targeting (tighter risk
control + better Sharpe/drawdown on the live strategies). Promoted to a
**validated, gate-passing, default-OFF option** in `quant/sizing/`.

**NOT yet live.** `quant/sizing/` is a shadow/backtest layer — it is NOT in the
live rebalance path (strategies vol-target internally). Actuating this requires a
further explicit human greenlight + a separate wiring step (either route the
sizing overlay into live rebalance, or wire the forecast into the strategies' own
vol-targeting). The default stays trailing/shadow until then.

## Live wiring (2026-06-10) — SHADOW, default-OFF

Wired as a **portfolio-level one-way overlay** in the rebalance path (Route B —
does NOT touch the strategies' internal sizing, so their validated evidence
stands). Mirrors the de-risk overlay exactly.

- `quant/live/voltarget.py`: `voltarget_multiplier(book_returns, cfg)` →
  `min(cap=1.0, trailing_vol / forecast_vol)` clamped to `floor=0.5`. **De-risk
  only** (cap 1.0 — never levers up; house style + one-way safety). `applied` =
  factor when `actuate` else 1.0 (shadow). Fail-safe: short/degenerate history ⇒
  1.0. `forecast_vol_ann_next` is the live one-day-ahead forecast.
- **Book vol proxy:** the live equity curve is days-long, far too short to fit a
  forecast — so the overlay reads the **allocation-weighted blend of the live
  strategies' OOS curves** (`_book_returns_for_voltarget`), the honest long proxy
  for the current book. (Switch to live history once it accrues 252 days.)
- `rebalance.py`: computes + reports the overlay, applies `voltarget.applied` to
  each strategy slice **alongside** de-risk (both ≤ 1.0 ⇒ compose one-way, the
  book can only ever be de-risked). CLI `--voltarget-actuate` (default shadow).
- Live dry-run today: forecast 12.6% > trailing 10.9% ⇒ computed **x0.87**,
  applied **x1.0 (SHADOW)** — 8 orders unchanged (byte-identical). 6 overlay
  unit-tests incl. fail-safe / de-risk-only / floor / shadow-gate.

**Status: SHADOW bake-in.** Actuation (`--voltarget-actuate`) is the final
explicit human greenlight after the overlay is observed across sessions — exactly
how the de-risk overlay was rolled out.
