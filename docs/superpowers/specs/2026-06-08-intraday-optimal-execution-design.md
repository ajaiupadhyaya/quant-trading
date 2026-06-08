# Intraday Optimal Execution (Almgren–Chriss) — Design Spec

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project A of the intraday/60s showcase track (depends on Sub-project 0, the spine, which is shipped).

---

## Context

The intraday spine (`quant/intraday/live/`) is live: a 60s loop trades a ring-fenced
ETF sleeve (QQQ/IWM/DIA) on Alpaca paper. Today each strategy entry is submitted as a
single marketable order on the tick it fires (clamped to the sleeve caps).

This sub-project adds **optimal execution**: instead of dumping an entry as one order,
the loop works it intraday along an Almgren–Chriss optimal trajectory, and the full
A–C machinery (efficient frontier + TWAP/VWAP/immediate baselines) is evaluated in the
existing intraday simulator.

**Goal = portfolio/learning showcase** (see [[project-quant-trading-intraday]]). Honest
caveat, stated to the owner and recorded here: the sleeve's per-trade cap (~$2k / ~20
shares of mega-liquid ETFs) means the *realized* impact savings on a live sleeve order
are negligible. The value is (a) the **technique** implemented rigorously and (b) the
**efficient-frontier analysis in the sim on realistic parent sizes**. Live wiring proves
the plumbing end-to-end; it will not move live P&L.

---

## Decisions locked in brainstorming

1. **Consumer:** wire into the **intraday sleeve** loop (touches the live paper sleeve;
   NOT the daily system, NOT the spine's guardrail ordering).
2. **Model depth:** **full Almgren–Chriss** — mean-variance optimal trajectory with a
   risk-aversion parameter λ, the efficient frontier (expected cost vs variance),
   permanent + temporary linear impact — **plus** TWAP / VWAP / immediate baselines.
3. **Safety boundary (§4):** only **entries** are worked along an A–C schedule. ALL
   de-risking — guardrail flattens, the daily-loss kill-switch, flat-by-close, and
   strategy-initiated exits — stays **immediate**. The spine's guardrails-first tick
   ordering is unchanged.
4. **No live learning:** A–C parameters (σ, η, γ) are computed from recent data at
   program-creation time; λ and horizon are config. Nothing trains from live fills.

---

## 1. Architecture

A new subpackage `quant/intraday/execution/` holding the A–C math and baselines as
**pure** functions, an `ExecutionProgram` that turns a planned trajectory into
per-tick child slices, and an `ExecutionManager` that the live loop uses to work
parent orders over multiple ticks. The same pure planner drives both the live loop
and the sim evaluation.

Rejected alternatives: baking the scheduler into the loop's submit step (untestable,
tangled); a sim-only adapter (doesn't satisfy "wire into the sleeve").

---

## 2. Components

`quant/intraday/execution/` (new subpackage):

| File | Responsibility |
|---|---|
| `almgren_chriss.py` | The A–C solver. Given parent shares `X`, horizon `T`, `N` intervals, volatility `σ`, temporary-impact `η`, permanent-impact `γ`, risk-aversion `λ` → the optimal **holdings trajectory** `x_0..x_N` and the per-interval **child sizes** `n_1..n_N`; plus closed-form **expected cost** `E[C]` and **variance** `V[C]`; and `efficient_frontier(lambda_grid) -> list[FrontierPoint]`. |
| `baselines.py` | `twap(X, N)` (equal slices), `vwap(X, volume_curve)` (volume-weighted slices), `immediate(X)` (one shot). Each returns a child-size schedule comparable to the A–C output. |
| `scheduler.py` | `ExecutionProgram`: wraps a chosen schedule for ONE parent order (symbol, side, total qty, tick horizon, start tick). `slice_due(tick_index) -> int` returns the child qty for the current tick; tracks `remaining`/`filled`; `is_complete`. Schedule-source-agnostic (A–C or any baseline). |
| `config.py` | `ExecConfig`: `horizon_ticks` (default 5), `risk_aversion` (λ, default chosen so the trajectory is moderately front-loaded), impact-coefficient source, σ-lookback. All overridable; no magic numbers. |
| `calibrate.py` | Compute `σ` from recent intraday returns and `(η, γ)` for a given symbol/parent. **Linear-vs-sqrt reconciliation:** the classic A–C closed form assumes *linear* temporary impact `h(v)=ηv`, but the repo's model (`quant/backtest/impact.py`, `market_impact_bps`/`trailing_dollar_adv`) is *square-root*. `calibrate.py` derives the linear `η` as a **local linearization of the sqrt model at the expected per-slice participation** (`η ≈ d(impact)/dv` evaluated at the planned slice size), so the solver stays closed-form while being anchored to the repo's real impact curve. `γ` (permanent) defaults to a small fraction of `η` (configurable). Returns `(σ, η, γ)`. The realized sim cost (§5) uses the true sqrt model, so the closed-form-vs-realized gap stays visible. |

---

## 3. The Almgren–Chriss model (what the solver computes)

Classic discrete A–C for working a parent of `X` shares over `N` equal intervals of
length `τ = T/N`:

- **Permanent impact** linear: `g(v) = γ v` (shifts the price permanently per share
  traded).
- **Temporary impact** linear: `h(v) = η v` (per-share penalty on the slice itself).
- **Optimal trajectory** for risk-aversion `λ`: the holdings follow
  `x_j = X · sinh(κ(T - t_j)) / sinh(κT)`, where `κ` solves
  `cosh(κτ) = 1 + (λ σ² τ²)/(2 η̃)` (with `η̃ = η − γτ/2`). Child sizes are the
  differences `n_j = x_{j-1} − x_j`.
- **Limits (test anchors):** `λ → 0` ⇒ `κ → 0` ⇒ the trajectory → the straight line
  (risk-neutral ⇒ TWAP-like equal liquidation); `λ → ∞` ⇒ strongly front-loaded
  (trade fast to kill risk).
- **Expected cost** `E[C]` and **variance** `V[C]` have closed forms in
  `(X, σ, η, γ, λ, τ, N)`; the **efficient frontier** is the locus of `(V[C], E[C])`
  as `λ` sweeps a grid — monotone (more risk-aversion ⇒ higher expected cost, lower
  variance).

All inputs are point-in-time; the solver is pure and deterministic.

---

## 4. Live integration — the `ExecutionManager`

A new `ExecutionManager` sits between strategy intent and order submission in the
sleeve loop:

- On a strategy **entry** intent, the manager calibrates `(σ, η, γ)` for the symbol,
  builds an A–C `ExecutionProgram` over `horizon_ticks`, and stores it keyed by symbol.
  (It does NOT submit the whole order immediately.)
- Each tick, the loop asks the manager for **due child slices** across active programs;
  each returned slice is submitted via `submit_simple_order` (unique COID) and is still
  subject to `clamp_qty_to_caps` and the trade-budget guard.
- A program is removed when complete (`filled == parent`) or cancelled.

**Immediate paths (NEVER scheduled) — the safety boundary:**
- Guardrail **flatten** (`_flatten_all` on loss-halt and flat-by-close) submits market
  orders immediately AND **cancels any active programs** for those symbols (a half-worked
  entry must not keep adding while we're trying to flatten).
- Strategy-initiated **exits** (reducing an existing position) submit immediately, as
  today, and also cancel any active entry program for that symbol.
- The spine's guardrails-first ordering (session → halt → quotes → loss-halt →
  flat-by-close → strategy) is unchanged; the manager only changes how *entry* orders
  reach the broker.

**Interaction with the mean-reversion strategy:** the strategy only opens when flat. If
an entry program for a symbol is still in-flight (not yet fully worked), the loop treats
the symbol as "position pending" so the strategy does not stack a second entry — i.e.
`position-or-active-program != 0` blocks a new open. (Prevents double-entry while a
program works.)

---

## 5. Evaluation (the showcase artifact)

- `quant/intraday/execution/evaluate.py`: an `IntradayStrategy` adapter that liquidates
  a fixed parent order via a chosen scheduler (A–C at a given λ, or a baseline) and is
  run through the **existing `BacktestEngine`** (`quant/intraday/sim/`) on historical
  intraday data to produce a **realized** execution cost (vs arrival price).
- CLI `quant intraday exec frontier --symbol QQQ --shares N [--horizon M]`: sweeps λ,
  prints the **efficient frontier** (expected cost vs variance from the closed form)
  alongside the **realized** sim costs of A–C vs TWAP vs VWAP vs immediate as labelled
  points. This is the headline deliverable that demonstrates the technique.
- CLI `quant intraday exec schedule --symbol QQQ --shares N`: prints the A–C child-size
  schedule for inspection.

---

## 6. Charter compliance

- **No lookahead:** σ/impact calibrated from data available at program start; the sim
  adapter is driven by the existing PIT replay.
- **Realistic execution:** evaluation runs through the real sim fill model
  (marketable far-touch + sqrt-impact + commission), not the A–C theoretical cost — the
  frontier reports BOTH the closed-form expectation and the realized sim cost so the gap
  is visible.
- **Overfitting guard:** the A–C model is closed-form with a single λ knob; no fitting.
- **Reproducibility:** pure deterministic solver, config-driven, CLI artifacts.
- **Honesty:** the spec and CLI output state plainly that live-sleeve impact savings are
  negligible at sleeve size; the technique's value is shown in the sim on larger sizes.

---

## 7. Success criteria

- The A–C solver reproduces the known limits (λ→0 ⇒ ~linear/TWAP trajectory; λ→∞ ⇒
  front-loaded) and its closed-form cost/variance match a numerical check; the efficient
  frontier is monotone.
- Baselines produce correct schedules (TWAP equal; VWAP weights ∝ volume curve, sum to
  parent; immediate = one slice).
- In the live sleeve loop, an entry is worked over multiple ticks via the manager, every
  child slice respects the sleeve caps, and a flatten / loss-halt / flat-by-close / exit
  mid-program cancels the remaining schedule and de-risks immediately.
- `quant intraday exec frontier` produces the frontier + baseline comparison artifact on
  real intraday data.
- The spine's existing tests and behavior are unchanged except for the documented
  entry-routing change; full suite stays green.

---

## 8. Out of scope / deferred

Nonlinear/transient impact models, adaptive (closed-loop) re-optimization mid-program,
limit-order placement tactics (this works marketable child orders), cross-asset
execution, RL-based execution (that is Sub-project C), and any change to the daily
system or to the spine's guardrail ordering.
