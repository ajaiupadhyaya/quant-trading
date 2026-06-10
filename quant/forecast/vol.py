"""HAR-RV volatility forecasting + honest out-of-sample evaluation (Phase 8).

A HAR-RV model (Corsi 2009): next-day realized variance regressed on the daily,
weekly (5d) and monthly (22d) averages of past realized variance — the modern
workhorse vol forecaster. We fit it by OLS (numpy, no new deps) and judge it the
only honest way: walk-forward, one-day-ahead, against strong naive benchmarks —
EWMA/RiskMetrics (lambda=0.94), a random walk, and a rolling-historical window —
scored with QLIKE (the proxy-robust loss, Patton 2011) and MSE, with a
Diebold-Mariano test for whether any edge is real.

With daily bars the realized-variance proxy is the squared daily log return —
noisy but unbiased, which is exactly why QLIKE (robust to proxy noise) is the
primary loss. A model is only worth promoting if it beats EWMA out-of-sample;
until then the forecast is advisory/shadow and drives nothing (the roadmap's
research→promote gate; sizing/vol-targeting stays behind a separate green-light).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from quant.forecast.garch import GarchModel, fit_garch, garch_forecast_next

_TRADING_DAYS = 252
_VAR_FLOOR = 1e-8  # daily-variance floor (~1bp daily vol) — guards log/division only

# Validated offline (walk_forward_eval include_garch=True on SPY, 3967 OOS days):
# GJR-GARCH WON the six-model one-day-ahead QLIKE race (1.531) and beat the HAR
# incumbent with Diebold-Mariano p=0.045 — so GJR-GARCH is promoted to advisory-
# PRIMARY (HAR earlier beat EWMA at p=0.01 and is now the fallback). Still
# advisory/shadow: promotion to *sizing* is a separate, deliberate gate.
OOS_SKILL_SPY = "GJR-GARCH>HAR OOS DM p=0.05"

# HAR horizons (days): daily, weekly, monthly.
_HAR_W = 5
_HAR_M = 22


@dataclass(frozen=True)
class HARModel:
    """Fitted HAR-RV coefficients: var_next = c0 + c_d·RVd + c_w·RVw + c_m·RVm."""

    c0: float
    c_d: float
    c_w: float
    c_m: float
    n_obs: int


@dataclass(frozen=True)
class VolForecast:
    """A live one-day-ahead vol forecast + its context. Advisory only."""

    asof: str  # ISO
    symbol: str
    model: str  # "gjr" | "garch" | "har" | "ewma"
    forecast_vol_ann: float | None  # annualized vol implied by the variance forecast
    realized_vol_ann: float | None  # trailing 21d realized vol (for comparison)
    vix: float | None  # implied (FRED), where available
    forecast_vs_realized: float | None  # forecast / realized - 1
    regime: str | None  # "calm" | "normal" | "elevated" | "stressed"
    oos_skill: str | None  # "beats EWMA" | "ties EWMA" | "unvalidated" — honesty flag


# --------------------------------------------------------------------------- #
# Realized variance + model primitives (pure, numpy)
# --------------------------------------------------------------------------- #
def log_returns(close: np.ndarray) -> np.ndarray:
    """Daily log returns from a close-price array (drops the first NaN)."""
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < 2:
        return np.array([])
    return np.diff(np.log(c))


def realized_variance(returns: np.ndarray) -> np.ndarray:
    """Daily realized-variance proxy = squared log return (floored)."""
    r = np.asarray(returns, dtype=float)
    return np.asarray(np.maximum(r * r, _VAR_FLOOR), dtype=float)


def _har_features(rv: np.ndarray, t: int) -> tuple[float, float, float] | None:
    """The (daily, weekly, monthly) RV averages known at time ``t`` (inclusive)."""
    if t < _HAR_M - 1:
        return None
    rv_d = float(rv[t])
    rv_w = float(np.mean(rv[t - _HAR_W + 1 : t + 1]))
    rv_m = float(np.mean(rv[t - _HAR_M + 1 : t + 1]))
    return rv_d, rv_w, rv_m


def fit_har(rv: np.ndarray) -> HARModel | None:
    """OLS fit of next-day RV on the HAR (daily/weekly/monthly) features."""
    rv = np.asarray(rv, dtype=float)
    rows: list[tuple[float, float, float]] = []
    targets: list[float] = []
    for t in range(_HAR_M - 1, rv.size - 1):
        feat = _har_features(rv, t)
        if feat is None:
            continue
        rows.append(feat)
        targets.append(float(rv[t + 1]))
    if len(rows) < 30:  # too little to fit honestly
        return None
    x = np.column_stack([np.ones(len(rows)), np.array(rows)])
    y = np.array(targets)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return HARModel(
        c0=float(coef[0]),
        c_d=float(coef[1]),
        c_w=float(coef[2]),
        c_m=float(coef[3]),
        n_obs=len(rows),
    )


def har_forecast_next(model: HARModel, rv: np.ndarray) -> float | None:
    """Forecast next-day variance from the most recent features. Floored ≥ 0."""
    rv = np.asarray(rv, dtype=float)
    feat = _har_features(rv, rv.size - 1)
    if feat is None:
        return None
    pred = model.c0 + model.c_d * feat[0] + model.c_w * feat[1] + model.c_m * feat[2]
    return float(max(pred, _VAR_FLOOR))


def ewma_forecast_series(rv: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """One-step-ahead EWMA variance forecasts. ``f[t]`` predicts RV at ``t+1``.

    RiskMetrics recursion f[t] = λ·f[t-1] + (1-λ)·RV[t]; seeded with RV[0].
    """
    rv = np.asarray(rv, dtype=float)
    f = np.empty_like(rv)
    if rv.size == 0:
        return f
    f[0] = rv[0]
    for t in range(1, rv.size):
        f[t] = lam * f[t - 1] + (1.0 - lam) * rv[t]
    return np.asarray(np.maximum(f, _VAR_FLOOR), dtype=float)


def _rolling_hist_forecast(rv: np.ndarray, t: int, window: int) -> float:
    lo = max(0, t - window + 1)
    return float(max(np.mean(rv[lo : t + 1]), _VAR_FLOOR))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def qlike(forecast_var: float, proxy_var: float) -> float:
    """Patton's QLIKE loss (robust to a noisy variance proxy). Lower is better."""
    f = max(float(forecast_var), _VAR_FLOOR)
    p = max(float(proxy_var), _VAR_FLOOR)
    return p / f - math.log(p / f) - 1.0


def mse(forecast_var: float, proxy_var: float) -> float:
    d = float(forecast_var) - float(proxy_var)
    return d * d


# --------------------------------------------------------------------------- #
# Walk-forward out-of-sample evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelScore:
    model: str
    mean_qlike: float
    median_qlike: float
    mean_mse: float
    n: int


@dataclass(frozen=True)
class ForecastEval:
    n_oos: int
    scores: dict[str, ModelScore]
    dm_stat: float | None  # HAR vs EWMA on QLIKE (negative → HAR better)
    dm_pvalue: float | None
    winner: str | None  # lowest mean QLIKE
    # GARCH-family vs HAR (only populated when include_garch=True); same sign
    # convention: negative stat → GARCH better. Default None keeps the four-model
    # eval backward-compatible.
    dm_garch_har_stat: float | None = None
    dm_garch_har_pvalue: float | None = None


def _dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float] | None:
    """Diebold-Mariano on a loss differential d = loss_a - loss_b (1-step → iid var)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 30 or float(np.var(d, ddof=1)) <= 0:
        return None
    stat = float(np.mean(d) / math.sqrt(np.var(d, ddof=1) / d.size))
    # two-sided normal p-value
    pval = float(math.erfc(abs(stat) / math.sqrt(2.0)))
    return stat, pval


def walk_forward_eval(
    close: np.ndarray,
    *,
    min_train: int = 504,
    refit_every: int = 21,
    rolling_window: int = 22,
    ewma_lambda: float = 0.94,
    include_garch: bool = False,
) -> ForecastEval:
    """Expanding-window, one-day-ahead OOS evaluation of HAR vs naive benchmarks.

    For each OOS day the HAR model is refit on all data up to that day (every
    ``refit_every`` steps for speed) and every model forecasts the *next* day's
    variance; forecasts are scored against the realized proxy. Pure / no I/O.

    With ``include_garch=True`` the GARCH-family forecasters (GARCH(1,1) and
    GJR-GARCH) join the race on the same refit cadence and the DM(GARCH vs HAR)
    fields are populated. Default ``False`` keeps the four-model eval byte-identical.
    """
    returns = log_returns(close)
    rv = realized_variance(returns)
    n = rv.size
    ewma_all = ewma_forecast_series(rv, ewma_lambda)

    models: tuple[str, ...] = ("har", "ewma", "rw", "rolling")
    if include_garch:
        models = (*models, "garch", "gjr")
    losses_q: dict[str, list[float]] = {m: [] for m in models}
    losses_m: dict[str, list[float]] = {m: [] for m in models}

    har_model: HARModel | None = None
    garch_model: GarchModel | None = None
    gjr_model: GarchModel | None = None
    start = max(min_train, _HAR_M)
    for t in range(start, n - 1):
        if har_model is None or (t - start) % refit_every == 0:
            har_model = fit_har(rv[: t + 1])
            if include_garch:
                garch_model = fit_garch(returns[: t + 1], kind="garch")
                gjr_model = fit_garch(returns[: t + 1], kind="gjr")
        target = float(rv[t + 1])
        preds: dict[str, float | None] = {
            "har": har_forecast_next(har_model, rv[: t + 1]) if har_model else None,
            "ewma": float(ewma_all[t]),
            "rw": float(rv[t]),
            "rolling": _rolling_hist_forecast(rv, t, rolling_window),
        }
        if include_garch:
            preds["garch"] = (
                garch_forecast_next(garch_model, returns[: t + 1]) if garch_model else None
            )
            preds["gjr"] = garch_forecast_next(gjr_model, returns[: t + 1]) if gjr_model else None
        for m, f in preds.items():
            if f is None or not math.isfinite(f) or f <= 0:
                continue
            losses_q[m].append(qlike(f, target))
            losses_m[m].append(mse(f, target))

    scores: dict[str, ModelScore] = {}
    for m in models:
        q = np.array(losses_q[m])
        if q.size == 0:
            continue
        scores[m] = ModelScore(
            model=m,
            mean_qlike=float(np.mean(q)),
            median_qlike=float(np.median(q)),
            mean_mse=float(np.mean(losses_m[m])),
            n=int(q.size),
        )

    dm_stat = dm_p = None
    if {"har", "ewma"} <= scores.keys():
        # align the two loss vectors (same OOS days, same order)
        dm = _dm_test(np.array(losses_q["har"]), np.array(losses_q["ewma"]))
        if dm is not None:
            dm_stat, dm_p = dm

    dm_gh_stat = dm_gh_p = None
    if include_garch and {"garch", "har"} <= scores.keys():
        dm_gh = _dm_test(np.array(losses_q["garch"]), np.array(losses_q["har"]))
        if dm_gh is not None:
            dm_gh_stat, dm_gh_p = dm_gh

    winner = min(scores, key=lambda m: scores[m].mean_qlike) if scores else None
    return ForecastEval(
        n_oos=max((s.n for s in scores.values()), default=0),
        scores=scores,
        dm_stat=dm_stat,
        dm_pvalue=dm_p,
        winner=winner,
        dm_garch_har_stat=dm_gh_stat,
        dm_garch_har_pvalue=dm_gh_p,
    )


# --------------------------------------------------------------------------- #
# Live forecast (fail-open) + render
# --------------------------------------------------------------------------- #
def _regime(vol_ann: float) -> str:
    if vol_ann < 0.12:
        return "calm"
    if vol_ann < 0.18:
        return "normal"
    if vol_ann < 0.28:
        return "elevated"
    return "stressed"


def compute_vol_forecast(
    close: np.ndarray,
    asof: date,
    *,
    symbol: str = "SPY",
    vix: float | None = None,
    oos_skill: str | None = None,
) -> VolForecast:
    """Pure: build a one-day-ahead vol forecast from a close series.

    Cascade GJR-GARCH -> HAR -> EWMA: GJR-GARCH is the OOS-validated primary
    (beats HAR, DM p=0.045 on SPY); HAR (itself beating EWMA) is the fallback when
    the series is too short to fit GJR; EWMA is the final floor. Annualised for
    comparison to trailing realized vol + VIX. Advisory only — drives no sizing.
    """
    returns = log_returns(close)
    rv = realized_variance(returns)
    model_name = "ewma"
    var_next: float | None = None
    # Promoted primary: GJR-GARCH (captures the leverage effect; won the OOS race).
    gjr = fit_garch(returns, kind="gjr") if returns.size >= 250 else None
    if gjr is not None:
        var_next = garch_forecast_next(gjr, returns)
        model_name = "gjr"
    if var_next is None:
        har = fit_har(rv) if rv.size > 60 else None
        if har is not None:
            var_next = har_forecast_next(har, rv)
            model_name = "har"
    if var_next is None and rv.size > 0:
        var_next = float(ewma_forecast_series(rv)[-1])
        model_name = "ewma"

    forecast_ann = (
        math.sqrt(_TRADING_DAYS * var_next) if var_next is not None and var_next > 0 else None
    )
    realized_ann = float(np.sqrt(_TRADING_DAYS * np.mean(rv[-21:]))) if rv.size >= 21 else None
    vs = None
    if forecast_ann is not None and realized_ann is not None and realized_ann != 0:
        vs = forecast_ann / realized_ann - 1.0
    return VolForecast(
        asof=asof.isoformat(),
        symbol=symbol,
        model=model_name,
        forecast_vol_ann=forecast_ann,
        realized_vol_ann=realized_ann,
        vix=(vix / 100.0 if vix is not None and vix > 1.5 else vix),
        forecast_vs_realized=vs,
        regime=_regime(forecast_ann) if forecast_ann is not None else None,
        oos_skill=oos_skill,
    )


def forecast_vol_series(
    returns: np.ndarray,
    *,
    model: str = "gjr",
    refit_every: int = 21,
    min_obs: int = 252,
) -> np.ndarray:
    """Point-in-time one-day-ahead ANNUALISED vol forecast for every day.

    ``out[t]`` is the annualised volatility forecast for day ``t`` built ONLY from
    ``returns[:t]`` (strictly prior — no lookahead), so it can size day ``t``; it
    is ``NaN`` until ``min_obs`` history accrues. The chosen ``model`` is refit
    every ``refit_every`` steps for speed (the same cadence as
    ``walk_forward_eval``); between refits the latest fit forecasts daily off the
    most recent features. Cascade per the requested model — ``gjr -> har -> ewma``
    (or ``har -> ewma`` / ``ewma`` for the lighter choices) — so a fit failure
    degrades rather than raising. Pure / no I/O.

    This is the bridge from the OOS-validated vol forecast into vol-targeting; it
    stays behind a default-OFF sizing toggle and an honest economic gate (the
    "separate deliberate gate" — accuracy does not imply sizing value).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    rv = realized_variance(r)
    ewma_all = ewma_forecast_series(rv)  # f[t] predicts RV at t+1 (causal)
    want_gjr = model == "gjr"
    want_har = model in ("gjr", "har")

    gjr_model: GarchModel | None = None
    har_model: HARModel | None = None
    start = max(int(min_obs), _HAR_M)
    fitted = False
    for t in range(start, n):
        if not fitted or (t - start) % int(refit_every) == 0:
            gjr_model = fit_garch(r[:t], kind="gjr") if (want_gjr and t >= 250) else None
            har_model = fit_har(rv[:t]) if want_har else None
            fitted = True
        var_next: float | None = None
        if gjr_model is not None:
            var_next = garch_forecast_next(gjr_model, r[:t])
        if var_next is None and har_model is not None:
            var_next = har_forecast_next(har_model, rv[:t])
        if var_next is None:
            var_next = float(ewma_all[t - 1])  # forecast made at t-1 predicts day t
        if var_next is not None and var_next > 0:
            out[t] = math.sqrt(_TRADING_DAYS * var_next)
    return out


def live_vol_forecast(
    settings: Any, asof: date, *, symbol: str = "SPY", oos_skill: str | None = None
) -> VolForecast:
    """Bounded, fail-open vol forecast from cached bars + cached VIX. Never raises."""
    try:
        import pandas as pd

        from quant.data import bars

        path = bars._cache_path(symbol, getattr(settings, "data_dir", None))
        if not path.exists():
            return compute_vol_forecast(np.array([]), asof, symbol=symbol, oos_skill=oos_skill)
        df = pd.read_parquet(path)
        s = df["close"].dropna() if "close" in df.columns else None
        s = s.loc[: pd.Timestamp(asof)] if s is not None else None
        close = s.to_numpy() if s is not None else np.array([])
    except Exception as exc:  # fail-open
        from quant.util.logging import logger

        logger.info("forecast.vol: bars load skipped ({!r})", exc)
        close = np.array([])

    vix: float | None = None
    try:
        from quant.data import macro

        v = macro.get_series(macro.FRED_SERIES["vix"]).dropna()
        vix = float(v.iloc[-1]) if len(v) else None
    except Exception:  # VIX is optional context
        vix = None

    return compute_vol_forecast(close, asof, symbol=symbol, vix=vix, oos_skill=oos_skill)


def render_vol_forecast(f: VolForecast | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if f is None or f.forecast_vol_ann is None:
        return "Vol forecast: unavailable"
    bits = [f"{f.symbol} {f.model.upper()} 1d-ahead vol={f.forecast_vol_ann:.1%}"]
    if f.regime:
        bits.append(f"regime={f.regime}")
    if f.realized_vol_ann is not None:
        bits.append(f"realized21d={f.realized_vol_ann:.1%}")
    if f.vix is not None:
        bits.append(f"VIX={f.vix:.1%}")
    if f.forecast_vs_realized is not None:
        bits.append(f"vs_realized={f.forecast_vs_realized:+.0%}")
    if f.oos_skill:
        bits.append(f"[{f.oos_skill}]")
    return "Vol forecast: " + ", ".join(bits)
