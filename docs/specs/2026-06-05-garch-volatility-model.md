# GARCH-family volatility modeling (charter gap #3)

**Date:** 2026-06-05
**Status:** shipped — GJR-GARCH wins the SPY OOS QLIKE race; GARCH beats HAR (DM p=0.045). Advisory/shadow.
**Charter tie:** CHARTER.md §"Open gaps", gap #3 — *ARIMA/GARCH volatility modeling
(techniques) — feeds vol-targeting / sizing.*

## Motivation

`quant/forecast/vol.py` already runs an honest, walk-forward, one-day-ahead
volatility horse-race: **HAR-RV** (Corsi 2009) vs naive benchmarks (EWMA /
RiskMetrics λ=0.94, random-walk, rolling-historical), scored with **QLIKE**
(Patton's proxy-robust loss) + MSE and a **Diebold-Mariano** test. A model is only
ever "advisory/shadow" until it beats the incumbent (EWMA) out-of-sample; it
drives no sizing until a separate green-light.

The one named technique the charter still calls missing is **GARCH**. GARCH(1,1)
(Bollerslev 1986) is the canonical conditional-variance model and the natural
member of this race — and the test fixture's data-generating process is literally
a GARCH(1,1), so a correct implementation should be *competitive by construction*
on that fixture, which doubles as a correctness signal.

We add two GARCH-family forecasters:

- **GARCH(1,1):** `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`
- **GJR-GARCH(1,1,1):** `σ²_t = ω + (α + γ·1[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}`
  — adds the **leverage/asymmetry** effect (down-moves raise vol more than
  up-moves), which is empirically strong in equity indices and is exactly what a
  plain GARCH(1,1) misses.

### Why not ARIMA (conditional mean) in this slice

Daily equity *returns* have essentially no linearly-predictable conditional mean
(an AR/ARIMA mean model on daily returns reliably **fails** to beat a random walk
OOS — that is the honest, expected finding). The charter's stated purpose for this
gap is "feeds vol-targeting / sizing," which is the **variance** process, not the
mean. So this slice delivers the GARCH family (the high-value, charter-purpose
piece) and **deliberately defers** ARIMA-mean modeling; if added later it belongs
in the factor/return-forecast path, gated by DSR like the GBM layer, not here.

## Design

New module `quant/forecast/garch.py`, pure numpy, no new dependency (no `arch`,
no scipy), fully deterministic given `(returns, kind)`:

- **Variance targeting** to cut the parameter count and stabilize the fit: pin the
  unconditional variance to the sample variance of demeaned returns, so
  `ω = (1 − α − β)·σ̄²` for GARCH and `ω = (1 − α − β − ½γ)·σ̄²` for GJR (½ from
  `E[1[ε<0]] = ½` under a symmetric innovation). We then optimize only the
  persistence parameters.
- **Deterministic QMLE:** minimize the Gaussian negative log-likelihood
  `½ Σ (log σ²_t + ε²_t/σ²_t)` over the stationarity simplex (`α,β,γ ≥ 0`,
  `α+β(+½γ) < 1`) by a **coarse grid → zoom-refine** search (no RNG, no scipy).
  Recursion seeded with `σ²_0 = σ̄²`.
- `fit_garch(returns, kind) -> GarchModel | None` (None on too-few/degenerate
  data → caller falls back, never raises in the live path).
- `garch_forecast_next(model, returns) -> float | None`: replay the conditional-
  variance recursion to T, return the one-step-ahead `σ²_{T+1}` (variance-floored).

### Integration — opt-in, behavior-preserving

`walk_forward_eval` gains `include_garch: bool = False`. **Default `False` keeps the
existing four-model race byte-identical** (regression-pinned); with `True` it adds
`"garch"` and `"gjr"` as competitors, refit on the same cadence as HAR. Two
backward-compatible optional fields are added to `ForecastEval`
(`dm_garch_har_stat`, `dm_garch_har_pvalue`, both default `None`) carrying the
DM(GARCH vs HAR) verdict.

`compute_vol_forecast` / `live_vol_forecast` are **unchanged** (HAR stays primary)
— GARCH is shadow/advisory via the eval command only, exactly like HAR was before
it earned promotion. CLI `quant forecast vol-eval` passes `include_garch=True` and
prints the GARCH/GJR rows plus the DM(GARCH vs HAR) line.

## Honesty / charter alignment

- One-day-ahead, expanding-window, QLIKE-primary, DM-significance — same bar HAR
  cleared. No model drives sizing from this change; it stays advisory.
- Default-OFF integration: zero change to any existing scored path, live forecast,
  or order/sizing logic until a conscious enable.
- Deterministic + dependency-free + PIT (only past returns enter each fit).

## Tests

- GARCH/GJR fit recovers a competitive forecast on the GARCH-simulated fixture
  (beats random-walk on QLIKE; α+β in (0,1)).
- One-step forecast is positive, finite, floored.
- `fit_garch` returns None on short/degenerate series (fail-soft).
- Determinism: same inputs → identical `GarchModel`.
- `walk_forward_eval(include_garch=False)` is unchanged vs today (regression);
  `include_garch=True` adds `garch`/`gjr` to `scores` and populates the DM fields.
- GJR's leverage term ≥ 0 and it ties-or-beats plain GARCH on a leverage-skewed
  series.
```
