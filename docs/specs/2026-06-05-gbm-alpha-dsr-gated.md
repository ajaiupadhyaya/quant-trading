# Gradient-boosting alpha layer (DSR-gated, research-only)

**Date:** 2026-06-05
**Status:** Design — building directly on `main` (single-main workflow).
**Charter tie-in:** closes gap #4 — "Gradient-boosting signal layer (techniques) —
strict OOS/DSR gating, given overfitting risk." Charter: *"ML where it's justified
(gradient boosting, not deep nets unless the data supports it)."*

## Problem & stance

There is no learned non-linear alpha model. The charter permits gradient boosting
*if the data supports it under strict OOS/DSR gating*, and explicitly warns about
overfitting. So the deliverable is **not** "a GBM that trades" — it is a
**gated evaluation**: a deterministic gradient-boosted regression-tree model
plugged into the *existing* purged walk-forward cross-sectional IC machinery
(`quant/forecast/factor.py`), with its out-of-sample long-short returns run
through the same **Deflated Sharpe / Probabilistic Sharpe** gates the live
strategies must pass. The honest verdict is reported as-is — large-cap factor
premia are modest and GBM overfits small panels, so a "GBM does not beat the
linear composite" result is an acceptable, expected outcome. **Nothing is wired
to any strategy, tilt, or allocation.**

## Why hand-rolled, no new dependency

The codebase hand-rolls every model from numpy (HMM, Kalman, HAR, ridge) for
determinism, PIT-safety, reproducibility, and interpretability — and ships **no
ML library**. A from-scratch gradient-boosted regression-tree learner (~200 LOC
numpy) keeps that invariant: fully deterministic (seeded), transparent, and
adds zero dependencies. Pulling in scikit-learn/lightgbm for one research model
would break the "no black-box" ethos the charter values.

## Design

### `quant/forecast/gbm.py` (pure, deterministic, no factor deps)

```
@dataclass(frozen=True)
class GBMConfig:
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 5
    subsample: float = 1.0        # row subsample per tree (seeded)
    seed: int = 0

fit_gbm(x: np.ndarray, y: np.ndarray, config) -> GBMModel
predict_gbm(model: GBMModel, x: np.ndarray) -> np.ndarray
```

- Squared-error regression trees, depth-limited, `min_samples_leaf`-bounded;
  exhaustive midpoint split search (small panels, cost is fine).
- Boosting: `F_0 = mean(y)`; each round fits a tree to residuals, adds
  `learning_rate * tree`. Optional seeded row subsample per round
  (`np.random.default_rng(seed + round)`).
- Deterministic for a given `(x, y, config)` — required for reproducibility and
  PIT-stable walk-forward.

### `quant/forecast/factor.py` — reuse the validation machinery

- New `_gbm_score_purged(loc, used_locs, panels, fwds, cfg)` mirroring
  `_ridge_score_purged`: assemble the embargo-purged past rebalances
  (`s + forward_days + embargo_days <= loc`), stack their z-scored factor panels
  (missing factor → 0, as ridge does) and demeaned forward returns, `fit_gbm`,
  then `predict_gbm` on the current `loc` panel → cross-sectional score.
- `walk_forward_factor_eval(..., model="gbm")` routes to it. **`composite`/`ridge`
  paths are untouched** (regression-pinned).
- `FactorConfig` gains a `gbm: GBMConfig` field (default `GBMConfig()`); ignored
  by non-gbm models.
- **Additive** `FactorEval` field `oos_spread_returns: tuple[float, ...] = ()`
  (the per-period top-minus-bottom tertile returns already computed — now
  retained so a return series exists for DSR/PSR). Existing fields/behavior
  unchanged.

### DSR/PSR gating — `gbm_research_verdict`

```
@dataclass(frozen=True)
class GBMVerdict:
    n_periods: int
    mean_rank_ic, rank_ic_tstat, mean_tertile_spread: float | None
    deflated_sharpe, probabilistic_sharpe: float | None
    passes_dsr, passes_psr, passes: bool
    note: str

gbm_research_verdict(closes, *, data_dir, config) -> GBMVerdict
```

- Runs `walk_forward_factor_eval` for the **model family** `{composite, ridge, gbm}`
  to get each model's OOS monthly L/S spread-return series.
- **Trial Sharpes** for deflation = each family member's per-period spread Sharpe
  (honest multiple-testing set: we tried 3 models). `deflated_sharpe(gbm_series,
  trial_sharpes)` + `probabilistic_sharpe(gbm_series, 0.0)`.
- Gates mirror live validation thresholds: `passes_dsr = DSR >= 0.30`,
  `passes_psr = PSR >= 0.70`, `passes = both`. The verdict is **observational** —
  it gates *promotion eligibility*, it does not promote.

### CLI — `quant forecast gbm-eval`

Mirrors `forecast factor-eval`: prints IC, tertile spread, DSR/PSR, and the
pass/fail verdict, with a research-only / survivorship-caveat banner. No artifact
is actuated.

## Safety / charter alignment

- **Zero live impact:** no strategy, tilt, allocation, or order path references
  the GBM. It is a CLI/eval-only research model, like `forecast vol-eval` /
  `factor-eval`.
- **Honest OOS:** purged + embargoed walk-forward; every prediction is out of
  sample; survivorship caveat stated (today's large-caps).
- **Overfitting-gated:** DSR deflates against the model family; PSR floor; the
  verdict reports failure plainly.
- **Deterministic & reproducible:** seeded, numpy-only, no deps.

## Out of scope (deferred)

- Promoting GBM to a live tilt/strategy (separate human-gated green-light, like
  every research→live promotion).
- Time-series (per-name) GBM, feature importinstance attribution, hyperparameter
  search (would itself need DSR trial-count accounting).

## Test plan (TDD)

GBM learner: fits a known non-linear function better than its mean baseline;
deterministic across runs; respects `max_depth`/`min_samples_leaf`; subsample
seed reproducible; constant-y → constant prediction; single-feature monotone
recovery.
Integration: `model="gbm"` runs the walk-forward on synthetic closes and returns
a populated `FactorEval` with `oos_spread_returns`; `gbm_research_verdict`
computes DSR/PSR and a coherent pass/fail; composite/ridge output byte-unchanged.

## Acceptance

- `quant/forecast/gbm.py` + `tests/forecast/test_gbm.py`; factor.py gbm branch +
  verdict + tests; CLI command.
- `ruff` + `mypy --strict` clean; full suite green; committed to `main`.
