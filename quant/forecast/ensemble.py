"""Validated stacking ensemble for forward-volatility forecasting (Phase 8).

The roadmap's "validated stacking combiner (purged CV) — no naive averaging." Two
of the Phase-8 models earned advisory status and both predict the SAME quantity —
forward market volatility — which is exactly what makes them stackable:

- the HAR-RV forecast (`quant/forecast/vol.py`, the current OOS champion), and
- the macro-conditioned regime crisis probability (`quant/forecast/regime.py`,
  whose crisis-prob has a validated forward-vol IC), plus the BOCPD change-point
  probability.

(The cross-sectional factor model is deliberately excluded — it was an OOS null,
and it predicts cross-sectional returns, not vol.)

The target is forward 21-day realized volatility. Base learners come in two
kinds: vol-unit forecasts (trailing-realized / EWMA / HAR — directly comparable
to the target) and orthogonal stress signals (regime crisis-prob, change-point
prob — in [0,1], contributing only through the learned combiner). The meta-learner
is non-negative least squares (interpretable convex-ish weights, no negative
loadings on a vol predictor), and — the part that makes it *validated* rather than
hopeful — it is judged by a NESTED, PURGED, walk-forward: at each test point the
combiner is fit only on rows ending an ``embargo`` (>= horizon) before it, so the
overlapping forward-vol windows cannot leak. The honest question: does the learned
stack beat the best single base learner AND a naive equal-weight average,
out-of-sample (QLIKE + Diebold-Mariano)? Stacking correlated forecasters often
does NOT — and if so this ships research-only, like the factor model.

numpy/scipy/pandas only — no new deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quant.forecast.vol import (
    _dm_test,
    ewma_forecast_series,
    fit_har,
    har_forecast_next,
    log_returns,
    mse,
    qlike,
    realized_variance,
)

_TRADING_DAYS = 252
_VAR_FLOOR = 1e-8

# Vol-unit base learners (comparable to the target) vs stress-signal learners
# (probabilities in [0,1] that only contribute through the combiner).
_VOL_LEARNERS = ("rw21", "ewma", "har")
_STRESS_LEARNERS = ("regime", "cp")
BASE_LEARNERS = _VOL_LEARNERS + _STRESS_LEARNERS

# Verdict from the honest nested walk-forward (walk_forward_stack, SPY 2005-2026,
# 4359 OOS points). HONEST NULL: mean QLIKE HAR 0.434 < naive-avg 0.447 < learned
# stack 0.504; the stack is significantly WORSE than HAR (DM +2.93, p=0.003) AND
# worse than a naive average (DM +5.92, p=3e-9). The combiner gives the regime
# crisis-prob ~0 weight (0.009) — it adds no incremental forward-vol *level*
# content beyond HAR (its validated value is crisis-prob *timing* as regime
# context, already wired advisory — a different use). So: RESEARCH-ONLY, HAR stays
# the advisory vol champion; NOT wired into MarketState/analyst. We deliberately do
# NOT search combiner variants until one wins (that is the spec-search the DSR
# one-way-trap forbids).
OOS_VERDICT: str | None = (
    "stack LOSES to HAR OOS (QLIKE +16%, DM p=0.003; < naive avg) — research-only"
)


@dataclass(frozen=True)
class StackConfig:
    horizon: int = 21  # forecast forward h-day realized vol
    embargo: int = 21  # purge rows whose target overlaps the test point (>= horizon)
    min_train: int = 504  # meta-learner minimum training rows
    refit_har_every: int = 21  # causal HAR refit cadence
    har_min_train: int = 504
    ridge_lambda: float = 0.0  # >0 → ridge meta-learner instead of NNLS


# --------------------------------------------------------------------------- #
# Base-learner panel (all causal / point-in-time)
# --------------------------------------------------------------------------- #
def _ann_vol(var: np.ndarray) -> np.ndarray:
    return np.asarray(np.sqrt(_TRADING_DAYS * np.maximum(var, _VAR_FLOOR)), dtype=float)


def _causal_har_var(rv: np.ndarray, *, min_train: int, refit_every: int) -> np.ndarray:
    """HAR 1-step variance forecast known at each t (refit every ``refit_every``).

    ``out[t]`` is the HAR forecast made using rv[0..t] (predicts t+1). NaN before
    enough history to fit. Causal by construction — never uses future data.
    """
    n = rv.size
    out = np.full(n, np.nan)
    model = None
    start = max(min_train, 22)
    for t in range(start, n):
        if model is None or (t - start) % refit_every == 0:
            fitted = fit_har(rv[: t + 1])
            if fitted is not None:
                model = fitted
        if model is not None:
            f = har_forecast_next(model, rv[: t + 1])
            if f is not None:
                out[t] = f
    return out


def forward_realized_vol(rv: np.ndarray, h: int) -> np.ndarray:
    """Annualized realized vol over the FUTURE window [t+1, t+h]. NaN off the end."""
    n = rv.size
    out = np.full(n, np.nan)
    for t in range(n - h):
        out[t] = math.sqrt(_TRADING_DAYS * max(float(np.mean(rv[t + 1 : t + 1 + h])), _VAR_FLOOR))
    return out


def build_base_panel(
    close: pd.Series,
    *,
    p_crisis: pd.Series | None = None,
    cp_prob: pd.Series | None = None,
    config: StackConfig | None = None,
) -> pd.DataFrame:
    """Assemble the causal base-learner forecasts + the forward-vol target.

    Columns: rw21 / ewma / har (annualized-vol forecasts), regime / cp (stress
    probabilities, optional — filled 0 when absent), target (forward h-day vol).
    Indexed by date. All learners are point-in-time; the target is strictly
    forward. Warm-up rows (vol learners NaN) are dropped, but the recent rows whose
    forward target is not yet known are KEPT (target NaN) so the live path can
    predict today; the eval/train paths restrict to finite-target rows themselves.
    """
    cfg = config or StackConfig()
    s = close.sort_index().astype(float)
    rv = realized_variance(log_returns(s.to_numpy()))
    # log_returns drops one row; align the index to the returns/rv vector.
    idx = s.index[1:]

    rw_var = pd.Series(rv, index=idx).rolling(cfg.horizon, min_periods=cfg.horizon).mean()
    ewma_var = ewma_forecast_series(rv)
    har_var = _causal_har_var(rv, min_train=cfg.har_min_train, refit_every=cfg.refit_har_every)
    target = forward_realized_vol(rv, cfg.horizon)

    panel = pd.DataFrame(
        {
            "rw21": _ann_vol(rw_var.to_numpy()),
            "ewma": _ann_vol(ewma_var),
            "har": _ann_vol(har_var),
            "target": target,
        },
        index=idx,
    )
    panel["regime"] = p_crisis.reindex(idx).astype(float) if p_crisis is not None else 0.0
    panel["cp"] = cp_prob.reindex(idx).astype(float) if cp_prob is not None else 0.0
    panel["regime"] = panel["regime"].fillna(0.0)
    panel["cp"] = panel["cp"].fillna(0.0)
    # Drop warm-up (vol learners NaN); keep recent rows with an unknown target.
    panel = panel.dropna(subset=["rw21", "ewma", "har"])
    return panel[["rw21", "ewma", "har", "regime", "cp", "target"]]


# --------------------------------------------------------------------------- #
# Meta-learner
# --------------------------------------------------------------------------- #
def fit_stacker(x: np.ndarray, y: np.ndarray, config: StackConfig | None = None) -> np.ndarray:
    """Fit the combiner weights. NNLS by default (non-negative loadings); ridge if
    ``ridge_lambda > 0``. Returns a weight vector aligned to the columns of ``x``."""
    cfg = config or StackConfig()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if cfg.ridge_lambda > 0:
        n_features = x.shape[1]
        a = x.T @ x + cfg.ridge_lambda * np.eye(n_features)
        coef = np.linalg.solve(a, x.T @ y)
        return np.asarray(coef, dtype=float)
    try:
        from scipy.optimize import nnls  # type: ignore[import-untyped]

        w, _ = nnls(x, y)
        return np.asarray(w, dtype=float)
    except Exception:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        return np.asarray(np.maximum(coef, 0.0), dtype=float)


@dataclass(frozen=True)
class StackEval:
    """Honest nested-walk-forward scorecard for the stacking ensemble."""

    n_oos: int
    mean_qlike: dict[str, float]  # per base learner + 'eq3' + 'stack'
    mean_mse: dict[str, float]
    best_base: str | None  # lowest-QLIKE single base learner among vol learners
    dm_stack_vs_best: tuple[float, float] | None  # (stat, pvalue); stat<0 → stack better
    dm_stack_vs_eq3: tuple[float, float] | None
    avg_weights: dict[str, float]  # mean learned stacker weight per learner
    verdict: str


def walk_forward_stack(panel: pd.DataFrame, config: StackConfig | None = None) -> StackEval:
    """Nested, purged, walk-forward evaluation of the stack vs its components.

    At each test row ``t`` the combiner is fit on rows ``[0, t - embargo)`` only
    (the embargo purges training targets whose forward window overlaps ``t``), then
    predicts ``t``. Scored against the single base learners, the naive equal-weight
    average of the vol learners, and via Diebold-Mariano. Pure / no I/O.
    """
    cfg = config or StackConfig()
    cols = list(BASE_LEARNERS)
    panel = panel[np.isfinite(panel["target"].to_numpy())]
    x = panel[cols].to_numpy(dtype=float)
    y = panel["target"].to_numpy(dtype=float)
    vol_idx = [cols.index(c) for c in _VOL_LEARNERS]
    n = len(panel)

    series_q: dict[str, list[float]] = {c: [] for c in (*cols, "eq3", "stack")}
    series_m: dict[str, list[float]] = {c: [] for c in (*cols, "eq3", "stack")}
    weights: list[np.ndarray] = []

    for t in range(cfg.min_train, n):
        train_end = t - cfg.embargo
        if train_end < 50:
            continue
        w = fit_stacker(x[:train_end], y[:train_end], cfg)
        weights.append(w)
        xt = x[t]
        target = float(y[t])
        preds: dict[str, float] = {c: float(xt[i]) for i, c in enumerate(cols)}
        preds["eq3"] = float(np.mean([xt[i] for i in vol_idx]))
        preds["stack"] = float(xt @ w)
        for name, f in preds.items():
            # score in VARIANCE space (QLIKE/MSE are variance losses)
            fv = max(f * f / _TRADING_DAYS, _VAR_FLOOR)
            tv = max(target * target / _TRADING_DAYS, _VAR_FLOOR)
            series_q[name].append(qlike(fv, tv))
            series_m[name].append(mse(fv, tv))

    mean_q = {k: float(np.mean(v)) for k, v in series_q.items() if v}
    mean_m = {k: float(np.mean(v)) for k, v in series_m.items() if v}
    best_base = min(_VOL_LEARNERS, key=lambda c: mean_q.get(c, math.inf)) if mean_q else None

    dm_best = (
        _dm_test(np.array(series_q["stack"]), np.array(series_q[best_base]))
        if best_base and series_q["stack"]
        else None
    )
    dm_eq3 = (
        _dm_test(np.array(series_q["stack"]), np.array(series_q["eq3"]))
        if series_q["stack"]
        else None
    )
    avg_w = (
        {c: float(np.mean([w[i] for w in weights])) for i, c in enumerate(cols)} if weights else {}
    )
    verdict = _verdict(mean_q, best_base, dm_best)
    return StackEval(
        n_oos=len(series_q["stack"]),
        mean_qlike=mean_q,
        mean_mse=mean_m,
        best_base=best_base,
        dm_stack_vs_best=dm_best,
        dm_stack_vs_eq3=dm_eq3,
        avg_weights=avg_w,
        verdict=verdict,
    )


def _verdict(
    mean_q: dict[str, float], best_base: str | None, dm_best: tuple[float, float] | None
) -> str:
    """Honest call: does the learned stack beat the best single base learner OOS?"""
    if not mean_q or best_base is None or "stack" not in mean_q:
        return "inconclusive"
    impr = mean_q[best_base] - mean_q["stack"]  # positive → stack lower QLIKE = better
    rel = impr / mean_q[best_base] if mean_q[best_base] else 0.0
    sig = dm_best is not None and dm_best[0] < 0 and dm_best[1] < 0.05
    if impr > 0 and sig and dm_best is not None:
        return (
            f"stack BEATS best-base {best_base} OOS (QLIKE {rel:.1%} lower, DM p={dm_best[1]:.3f})"
        )
    if impr > 0:
        return f"stack edges {best_base} (QLIKE {rel:.1%} lower) but NOT DM-significant — research-only"
    return (
        f"stack does NOT beat {best_base} OOS (QLIKE {-rel:.1%} higher) — research-only, keep HAR"
    )


# --------------------------------------------------------------------------- #
# Live read (fail-open) + render
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StackForecast:
    """Live stacked forward-vol forecast + the learned weights. Advisory only."""

    asof: str
    horizon: int
    forecast_vol_ann: float | None
    base: dict[str, float] = field(default_factory=dict)  # current base-learner values
    weights: dict[str, float] = field(default_factory=dict)
    oos_verdict: str | None = None


def compute_stack(panel: pd.DataFrame, config: StackConfig | None = None) -> StackForecast | None:
    """Fit the combiner on all rows with a known target, predict the latest row.

    The most recent ``horizon`` rows have an unknown (future) target so they are
    excluded from fitting; their *base* values still exist, so the latest row (with
    target NaN) is the one we forecast.
    """
    cfg = config or StackConfig()
    if panel.empty:
        return None
    cols = list(BASE_LEARNERS)
    train = panel.dropna(subset=["target"])
    if len(train) < cfg.min_train // 2:
        return None
    w = fit_stacker(train[cols].to_numpy(float), train["target"].to_numpy(float), cfg)
    last = panel.iloc[-1]
    xt = last[cols].to_numpy(dtype=float)
    fc = float(xt @ w)
    return StackForecast(
        asof=str(panel.index[-1].date()),
        horizon=cfg.horizon,
        forecast_vol_ann=fc if math.isfinite(fc) and fc > 0 else None,
        base={c: float(last[c]) for c in cols},
        weights={c: float(w[i]) for i, c in enumerate(cols)},
        oos_verdict=OOS_VERDICT,
    )


def _live_regime_series(
    settings: Any, close: pd.Series, asof: date
) -> tuple[pd.Series | None, pd.Series | None]:
    """Best-effort causal regime crisis-prob + change-point series for the live stack."""
    try:
        from quant.data import macro
        from quant.forecast.regime import (
            MacroRegimeConfig,
            build_macro_regime_features,
            change_point_series,
        )
        from quant.regime.detect import DetectConfig, run_detection

        feats = build_macro_regime_features(
            spy_close=close,
            vix=macro.get_series(macro.FRED_SERIES["vix"]),
            dgs10=macro.get_series(macro.FRED_SERIES["tenyear"]),
            dgs2=macro.get_series(macro.FRED_SERIES["twoyear"]),
            baa=macro.get_series(macro.FRED_SERIES["baa"]),
            aaa=macro.get_series(macro.FRED_SERIES["aaa"]),
            macro_config=MacroRegimeConfig(use_credit=True),
        )
        # Bounded config for a live read (quarterly refit, 3y window).
        series = run_detection(
            feats, DetectConfig(refit_freq="QS", train_window_days=252 * 3, n_restarts=3)
        )
        p_crisis = series["p_crisis"]
        ret = (
            pd.Series(
                np.log(close.sort_index().astype(float).to_numpy()), index=close.sort_index().index
            )
            .diff()
            .dropna()
        )
        cp = change_point_series(ret)["cp_prob"]
        return p_crisis, cp
    except Exception:
        return None, None


def live_stack(
    settings: Any, asof: date, *, symbol: str = "SPY", config: StackConfig | None = None
) -> StackForecast | None:
    """Bounded, fail-open live stacked forward-vol forecast. Never raises."""
    cfg = config or StackConfig()
    try:
        import pandas as pd

        from quant.data import bars

        path = bars._cache_path(symbol, getattr(settings, "data_dir", None))
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        s = df["close"].dropna().loc[: pd.Timestamp(asof)]
        if len(s) < cfg.min_train:
            return None
        p_crisis, cp = _live_regime_series(settings, s, asof)
        panel = build_base_panel(s, p_crisis=p_crisis, cp_prob=cp, config=cfg)
        return compute_stack(panel, cfg)
    except Exception as exc:  # fail-open
        from quant.util.logging import logger

        logger.info("forecast.ensemble: live stack skipped ({!r})", exc)
        return None


def render_stack(f: StackForecast | None) -> str:
    """Terse one-liner for the CLI / logs / (if promoted) the Claude prompt."""
    if f is None or f.forecast_vol_ann is None:
        return "Vol ensemble: unavailable"
    top = sorted(f.weights.items(), key=lambda kv: -kv[1])[:3]
    wbits = ", ".join(f"{k}={v:.2f}" for k, v in top if v > 1e-6)
    bits = [f"{f.horizon}d-fwd vol={f.forecast_vol_ann:.1%}", f"weights[{wbits}]"]
    if f.oos_verdict:
        bits.append(f"[{f.oos_verdict}]")
    return "Vol ensemble: " + ", ".join(bits)
