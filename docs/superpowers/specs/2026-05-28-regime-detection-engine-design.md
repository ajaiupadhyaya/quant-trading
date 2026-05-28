# Regime Detection Engine — Design

**Date:** 2026-05-28
**Status:** Approved (brainstorm), pending implementation plan
**Author:** ajaiupadhyaya + Claude

## Context

`quant-trading` is a mature evidence-gated paper-trading platform: full
backtest/validation battery (walk-forward, CPCV, DSR, PSR, stationary-block
bootstrap, regime stress, cost sweep), six ETF/equity strategies, governance +
evidence gating, live Alpaca execution, reconciliation, halt/resume, a pretrade
risk primitive, and a Textual TUI.

A longer-term vision (from the user's `quant-resources-1` lecture corpus) is an
autonomous, terminal-monitored quant system built on four pillars: a **regime
detector (HMM/Kalman)**, a **strategy selector**, an **execution/risk engine
(Greeks, Kelly, drawdown throttles)**, and an **autonomous monitoring daemon
(heartbeat + alerts + kill-switch)**. Those pillars are mostly missing or only
partial today. Each is its own spec → plan → build cycle.

This spec covers **only the first pillar: the regime detection engine.** The
other three are explicitly out of scope here and become follow-on specs.

### What exists today (and is reused, not replaced)

- `quant/backtest/regimes.py` — hard-coded historical crisis windows
  (GFC-2008, China-2015, COVID-2020, bear-2022, bull-2024). Reused as
  **ground-truth labels** for validating the detector.
- `quant/strategies/_regime_overlay.py` — a *rule-based* live de-risk overlay
  (SPY 200d SMA + VIX + strategy-equity SMA). Stays as the live de-risk
  mechanism **until the HMM signal graduates**. Not touched by this work.
- `quant/strategies/_kalman.py` — a hand-rolled Kalman recursion for pairs
  hedge ratios. Establishes the numpy-only, mypy-strict style this work follows.
- `quant/data/macro.py` — FRED fetcher (VIX `VIXCLS`, DGS10, DGS2, …).
- `quant/data/bars.py` — yfinance OHLCV cache for SPY + the ETF universe.
- `quant/research/registry.py` — append-only experiment registry.
- `quant/tui.py` — `quant monitor` Textual dashboard.

## Decisions (locked during brainstorm)

1. **First pillar:** regime detection engine (foundation the other pillars consume).
2. **Initial role:** an **observed, gated signal** — it labels the market state,
   logs it, and surfaces it, but does **not** change any live position until it
   passes its own validation gate. Wiring into allocation is a follow-on spec.
3. **Granularity:** a single **market-wide** regime (not per-asset).
4. **Taxonomy:** **3 states** interpreted post-hoc as `calm-bull` / `choppy` /
   `crisis` (risk-on / neutral / risk-off).
5. **Implementation:** a **hand-rolled Gaussian HMM in numpy/scipy** plus a
   Kalman feature smoother. Zero new heavy dependencies; mypy-strict clean.

## Non-negotiable: point-in-time correctness

Baum-Welch fits on the *entire* sample, and the smoothed (forward-backward)
state path uses future data — both are look-ahead. Live regime labels MUST come
from a **walk-forward refit + forward-*filtered* posteriors** that condition only
on data up to time *t*. The full-sample smoothed/Viterbi path is permitted
**only** for offline backtest visualization and is always flagged in-sample.
This discipline is enforced by an automated truncation test (see Validation,
gate 4).

## Architecture & module layout

New self-contained package `quant/regime/`:

```
quant/regime/
  models.py       # frozen dataclasses: HMMParams, RegimeSeries, RegimeReport
  features.py     # PIT market feature matrix from bars + macro
  kalman_state.py # 1-D local-level Kalman smoother for trend/vol features
  hmm.py          # 3-state Gaussian HMM: fit (Baum-Welch), filter, Viterbi, score
  detect.py       # walk-forward PIT refit loop → daily filtered posteriors + canonical labels
  validation.py   # regime-signal validation gate (metrics + pass/fail)
```

**Unit boundaries:**

- `hmm.py` is pure math: it takes a `(T, K)` float array and returns
  posteriors / params. It knows nothing about markets, dates, or files. Fully
  testable against synthetic data from a known HMM.
- `features.py` is pure data transformation: bars + macro series in, an aligned
  PIT feature frame out.
- `kalman_state.py` is a pure online smoother (numpy in, numpy out).
- `detect.py` is the **only** orchestrator: refit cadence, state→label
  identification, persistence. Depends on the other three.
- `validation.py` consumes a `RegimeSeries` + ground-truth windows and emits a
  `RegimeReport`.

Integration touchpoints (existing code, additive only): a `quant regime` CLI
group in `cli.py`; experiment logging via `research/registry.py`; a regime
panel in `tui.py`; artifacts under `data/regime/`.

## The model

### Gaussian HMM (`hmm.py`)

3 hidden states, `K` Gaussian-emission features. Parameters
(`HMMParams`, frozen): initial `pi (3,)`, transition `A (3,3)`, emission means
`mu (3,K)`, diagonal covariances `var (3,K)`.

- **fit** — Baum-Welch EM in log-space (`scipy.special.logsumexp`).
  **Multiple seeded random restarts**, keep the best log-likelihood (EM is
  non-convex). Diagonal covariances with `eps * I` regularization to prevent
  state collapse / singular covariance. Convergence on log-likelihood delta with
  a max-iteration cap. RNG seeded so `(git SHA, seed)` reproduces a fit.
- **filter** (`forward`) — online α-recursion returning
  `P(state_t | obs_1..t)`. **The only thing live decisions ever consume.**
- **viterbi** — most-likely full path, for offline backtest labeling/plots only,
  explicitly in-sample (uses future data; never live).
- **score** — total log-likelihood, for restart selection and diagnostics.

Numerical guards: all recursions in log-space; covariance floor; degenerate /
too-short input returns a typed error or `None` (consistent with
`_kalman.py`); finite-value assertions.

### Kalman feature smoother (`kalman_state.py`)

A 1-D local-level (random-walk-plus-noise) Kalman filter that denoises the trend
and realized-vol features **before** they enter the HMM, reducing label
whipsaw. Online by construction (no look-ahead). Process/observation variances
are exposed config parameters. Reuses the recursion pattern proven in
`strategies/_kalman.py`.

## Features & data flow (`features.py`)

Market-wide daily feature matrix, all from local data:

| Feature | Source | Captures |
|---|---|---|
| SPY log return (Kalman-smoothed) | bars | direction / trend |
| SPY realized vol, 21d (log) | bars | turbulence |
| VIX level | FRED `VIXCLS` | forward-looking risk |
| SPY drawdown from trailing peak | bars | crisis proxy |
| Term spread (DGS10 − DGS2) *(optional, config flag, default on)* | FRED | macro stress |

**Look-ahead discipline:**

1. **Trailing-only standardization** — features are z-scored against an
   expanding (or rolling-window) mean+std as of each day, never full-sample.
2. **Walk-forward refit** — the HMM refits on a schedule (default: monthly) over
   a rolling 5y window (expanding window is a config option). Between refits,
   params are frozen and only `forward` runs on new days.
3. **Canonical state identification** — raw EM state indices are arbitrary and
   reshuffle each refit. After every refit, states are relabeled to
   `{calm-bull, choppy, crisis}` by ranking on their fitted (mean-return, vol)
   signature, so the daily label series stays continuous across refit
   boundaries. Tie-breaking rule is deterministic.
4. **Calendar alignment** — bars (NYSE trading days) and FRED series (which can
   lag / have gaps) are aligned with forward-fill of the macro series only up to
   the as-of day; no backfill.

**Artifacts:**

- `data/regime/regime_series.parquet` — daily `[p_calm, p_choppy, p_crisis]`
  filtered posteriors + argmax hard label + refit-epoch id.
- `data/regime/model.json` — latest fitted `HMMParams` + fit metadata
  (window, seed, git SHA, log-likelihood, restart count).

## Validation gate (`validation.py`)

The signal graduates from "observed" to "tradable" only by passing four gates,
all out-of-sample / walk-forward (uses the *filtered* series, never smoothed):

1. **Persistence** — median regime duration is economically plausible (no daily
   flip-flopping); excessive churn fails.
2. **Economic coherence** — the `crisis` state overlaps the hard-coded
   historical crisis windows in `backtest/regimes.py` better than a
   frequency-matched random baseline, AND realized *forward* vol increases
   monotonically `calm` → `choppy` → `crisis`.
3. **Predictive lift** — using *yesterday's filtered* label to de-risk a simple
   SPY (or strategy) overlay improves a tail/risk metric (max-drawdown reduction
   or DSR lift) out-of-sample only. The always-on book is the baseline.
4. **No-look-ahead audit** — truncation test: the filtered label at day *t* is
   identical whether or not days *t+1…* exist in the input.

Pass → logged as a validated signal in the research registry with a
`validation_report.json`-style sidecar. Failing gate(s) are reported with the
specific shortfall. **Wiring into live allocation is deliberately deferred to a
follow-on spec.**

## Integration

- **CLI** — a `quant regime` group in `cli.py`:
  - `quant regime fit` — refit on the configured window, persist params +
    series, append a registry experiment.
  - `quant regime label [--asof DATE]` — print the current (or historical)
    filtered posterior + hard state.
  - `quant regime backtest` — Viterbi path + per-regime stats + plot
    (in-sample, labeled as such).
  - `quant regime validate` — run the four gates, write the sidecar, append a
    registry experiment.
- **Research registry** — each `fit` / `validate` appended with git SHA, params,
  seed, metrics, and gate results (reuses `research/registry.py`).
- **TUI** — a regime panel in `quant monitor`: current state, posterior bar,
  days-in-state. Built behind the existing snapshot pattern so it is testable
  without the Textual event loop.

## Testing

- **HMM recovery** — generate data from a known HMM; assert EM recovers params
  within tolerance; `forward` filtered posteriors match an independent reference
  filtered computation (and are a valid prefix of the smoothed marginals);
  `viterbi` recovers the known path at low noise.
- **Numerical** — log-space stability, covariance-floor prevents collapse,
  degenerate / short input handled, seeded reproducibility (same seed → same
  params).
- **PIT** — the truncation test (gate 4) as a standalone unit test.
- **Features** — trailing-only standardization (no full-sample leakage), no NaN
  leakage, bars/macro calendar alignment, forward-fill-not-backfill on macro.
- **State identification** — canonical relabel is deterministic and stable
  across refits on a fixture.
- **Property tests** (hypothesis, already a dev dep) — transition rows sum to 1,
  filtered posteriors sum to 1, labels in `{0,1,2}`, parameters finite.
- **CLI** — each command runs end-to-end on a small fixture (no network).

## Out of scope (explicit)

- Wiring the HMM regime into `_regime_overlay.py` or live allocation (follow-on).
- Per-asset regimes; >3 states; non-Gaussian emissions.
- Adding `hmmlearn` / `statsmodels` / `filterpy` dependencies.
- The other three pillars (strategy selector, options/Greeks engine, autonomous
  monitoring daemon + kill-switch).

## Dependencies

No new runtime dependencies — numpy + scipy (`scipy.special.logsumexp`) only,
both already present. `hypothesis` and `matplotlib` (for the backtest plot) are
already in the project.
