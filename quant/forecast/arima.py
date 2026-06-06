"""Deterministic ARIMA / ARMA conditional-mean modeling (pure numpy, no deps).

Completes the charter's named technique set (ARIMA/GARCH) — GARCH is the variance
half (:mod:`quant.forecast.garch`); this is the **mean** half. Hand-rolled
Hannan-Rissanen two-stage OLS ARMA(p,q) with differencing ``d``, fully
deterministic given ``(y, config)``, transparent, PIT-stable, dependency-free (no
``statsmodels``).

It exists to *establish*, not assert, the efficient-market prior for the daily
conditional mean: fit the canonical model, evaluate it the only honest way
(walk-forward, one-step-ahead, DSR/PSR-gated against the no-predictability
benchmark), and let the negative result stand on the record. That documented "no
edge" is the empirical justification for the system's architecture — it forecasts
variance (GARCH) and the cross-section (factors), *not* the daily mean, because the
daily mean is unforecastable. The same machinery would detect a real edge if one
existed (verified on a synthetic AR(1)), so this is a true test. Research-only —
wired to no strategy, tilt, sizing, or order path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_OBS = 250  # too little to fit an ARMA structure honestly


@dataclass(frozen=True)
class ARIMAConfig:
    p: int = 1  # AR order
    d: int = 0  # differencing order (0 or 1 supported)
    q: int = 1  # MA order
    seed_ar_order: int = 0  # Hannan-Rissanen stage-1 AR order (0 -> auto)


@dataclass(frozen=True)
class ARIMAModel:
    phi: tuple[float, ...]  # AR coefficients (length p)
    theta: tuple[float, ...]  # MA coefficients (length q)
    d: int
    mean: float  # mean of the differenced series
    n_obs: int


def _difference(y: np.ndarray, d: int) -> np.ndarray:
    z = np.asarray(y, dtype=float)
    for _ in range(d):
        z = np.diff(z)
    return z


def _arma_residuals(zc: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """In-sample residuals of a zero-mean ARMA(p,q) filter (recursive)."""
    n = zc.size
    p, q = phi.size, theta.size
    e = np.zeros(n, dtype=float)
    for t in range(n):
        ar = 0.0
        for i in range(p):
            if t - 1 - i >= 0:
                ar += phi[i] * zc[t - 1 - i]
        ma = 0.0
        for j in range(q):
            if t - 1 - j >= 0:
                ma += theta[j] * e[t - 1 - j]
        e[t] = zc[t] - ar - ma
    return e


def _invertible(theta: np.ndarray) -> np.ndarray:
    """Rescale MA coefficients so ``sum|theta| < 1`` — keeps the residual recursion
    bounded (a sufficient invertibility safeguard)."""
    s = float(np.abs(theta).sum())
    return theta * (0.99 / s) if s >= 1.0 else theta


def fit_arima(y: np.ndarray, config: ARIMAConfig | None = None) -> ARIMAModel | None:
    """Fit ARIMA(p,d,q) by deterministic Hannan-Rissanen. None on too-few data."""
    cfg = config or ARIMAConfig()
    if cfg.d not in (0, 1):
        return None
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < _MIN_OBS:
        return None
    z = _difference(arr, cfg.d)
    if z.size < _MIN_OBS // 2:
        return None
    mu = float(z.mean())
    zc = z - mu
    n = zc.size
    p, q = cfg.p, cfg.q

    if q == 0:
        # Pure AR(p): a single OLS, no residual pre-whitening needed.
        if n - p < 5 * p + 10:
            return None
        rows = np.column_stack([zc[p - 1 - i : n - 1 - i] for i in range(p)])
        target = zc[p:]
        phi, *_ = np.linalg.lstsq(rows, target, rcond=None)
        return ARIMAModel(
            phi=tuple(float(c) for c in phi),
            theta=(),
            d=cfg.d,
            mean=mu,
            n_obs=int(arr.size),
        )

    # Hannan-Rissanen stage 1: long AR(m) to recover residual estimates.
    m = cfg.seed_ar_order or max(10, 2 * (p + q))
    m = min(m, n // 4)
    if m < p + q or n - m < 5 * (p + q) + 10:
        return None
    ar_rows = np.column_stack([zc[m - 1 - i : n - 1 - i] for i in range(m)])
    ar_coef, *_ = np.linalg.lstsq(ar_rows, zc[m:], rcond=None)
    ehat = np.zeros(n, dtype=float)
    for t in range(m, n):
        pred = float(sum(ar_coef[i] * zc[t - 1 - i] for i in range(m)))
        ehat[t] = zc[t] - pred

    # Stage 2: regress zc[t] on its p lags + q lagged residuals.
    start = max(p, m + q)
    if n - start < 3 * (p + q) + 10:
        return None
    design = np.empty((n - start, p + q), dtype=float)
    for k, t in enumerate(range(start, n)):
        design[k, :p] = [zc[t - 1 - i] for i in range(p)]
        design[k, p:] = [ehat[t - 1 - j] for j in range(q)]
    coef, *_ = np.linalg.lstsq(design, zc[start:], rcond=None)
    phi = coef[:p]
    theta = _invertible(coef[p:])
    return ARIMAModel(
        phi=tuple(float(c) for c in phi),
        theta=tuple(float(c) for c in theta),
        d=cfg.d,
        mean=mu,
        n_obs=int(arr.size),
    )


def arima_forecast_next(model: ARIMAModel, y: np.ndarray) -> float | None:
    """One-step-ahead forecast of ``y`` (original scale, integrating ``d``)."""
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    z = _difference(arr, model.d)
    if z.size == 0:
        return None
    zc = z - model.mean
    phi = np.asarray(model.phi, dtype=float)
    theta = np.asarray(model.theta, dtype=float)
    e = _arma_residuals(zc, phi, theta) if theta.size else np.zeros(zc.size)
    n = zc.size
    ar = float(sum(phi[i] * zc[n - 1 - i] for i in range(phi.size) if n - 1 - i >= 0))
    ma = float(sum(theta[j] * e[n - 1 - j] for j in range(theta.size) if n - 1 - j >= 0))
    z_next = model.mean + ar + ma
    forecast = float(arr[-1] + z_next) if model.d == 1 else float(z_next)
    return forecast if np.isfinite(forecast) else None


# --------------------------------------------------------------------------- #
# Honest walk-forward evaluation + DSR/PSR-gated verdict
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ARIMAEval:
    p: int
    q: int
    d: int
    n_oos: int
    mean_ic: float | None  # OOS corr of the conditional deviation vs demeaned realized
    ic_tstat: float | None
    hit_rate: float | None  # directional accuracy on the conditional signal
    mse_ratio: float | None  # model MSE / unconditional-baseline MSE (<1 = conditioning helps)
    oos_strategy_returns: tuple[float, ...]  # drift-neutral: sign(dev) * (realized - baseline)


def walk_forward_arima_eval(
    y: np.ndarray,
    *,
    config: ARIMAConfig | None = None,
    min_train: int = 504,
    refit_every: int = 21,
    cost_bps: float = 2.0,
) -> ARIMAEval:
    """Expanding-window, one-step-ahead OOS evaluation of an ARIMA mean forecast.

    Refits every ``refit_every`` steps; each step forecasts the next value and the
    (forecast, realized) pair is scored. The drift-neutral sign-strategy charges a
    realistic ``cost_bps`` per position flip — a daily-frequency mean signal that
    only "works" gross of costs must not look tradeable (charter principle 2).
    Pure / no I/O.
    """
    cfg = config or ARIMAConfig()
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    forecasts: list[float] = []
    realized: list[float] = []
    baselines: list[float] = []  # the model's UNCONDITIONAL one-step forecast
    model: ARIMAModel | None = None
    start = max(min_train, _MIN_OBS)
    for t in range(start, n - 1):
        if model is None or (t - start) % refit_every == 0:
            model = fit_arima(arr[: t + 1], cfg)
        if model is None:
            continue
        f = arima_forecast_next(model, arr[: t + 1])
        if f is None or not np.isfinite(f):
            continue
        # Unconditional baseline: what the model predicts with zero AR/MA signal.
        # d=0 -> the series mean; d=1 -> last level + mean increment. Subtracting it
        # isolates the *conditional* skill from the asset's drift, so the gated
        # series can't pass just by being permanently long a drifting market.
        baseline = model.mean + (float(arr[t]) if model.d == 1 else 0.0)
        forecasts.append(f)
        realized.append(float(arr[t + 1]))
        baselines.append(baseline)

    fa = np.asarray(forecasts, dtype=float)
    ra = np.asarray(realized, dtype=float)
    ba = np.asarray(baselines, dtype=float)
    dev = fa - ba  # conditional deviation
    exc = ra - ba  # demeaned realized
    # Drift-neutral sign-strategy, NET of a per-flip transaction cost: the position
    # is sign(deviation); a sign change from the prior step pays cost_bps.
    pos = np.sign(dev)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0  # entry counts as a flip
    cost = flips.astype(float) * (cost_bps / 1e4)
    strat = tuple(float(pos[i] * exc[i] - cost[i]) for i in range(dev.size))
    mean_ic = ic_t = hit = mse_ratio = None
    if dev.size >= 2 and float(dev.std()) > 0 and float(exc.std()) > 0:
        ic = float(np.corrcoef(dev, exc)[0, 1])
        mean_ic = ic
        if abs(ic) < 1.0:
            ic_t = float(ic * np.sqrt(dev.size - 2) / np.sqrt(1 - ic * ic))
        hit = float(np.mean(np.sign(dev) == np.sign(exc)))
        mse_base = float(np.mean(exc * exc))
        mse_model = float(np.mean((fa - ra) ** 2))
        mse_ratio = mse_model / mse_base if mse_base > 0 else None
    return ARIMAEval(
        p=cfg.p,
        q=cfg.q,
        d=cfg.d,
        n_oos=int(fa.size),
        mean_ic=mean_ic,
        ic_tstat=ic_t,
        hit_rate=hit,
        mse_ratio=mse_ratio,
        oos_strategy_returns=strat,
    )


@dataclass(frozen=True)
class ARIMAVerdict:
    best_p: int
    best_q: int
    n_oos: int
    mean_ic: float | None
    ic_tstat: float | None
    hit_rate: float | None
    mse_ratio: float | None
    deflated_sharpe: float | None
    probabilistic_sharpe: float | None
    passes_dsr: bool
    passes_psr: bool
    passes: bool
    note: str


_DSR_GATE = 0.30
_PSR_GATE = 0.70
_DEFAULT_GRID: tuple[tuple[int, int], ...] = ((1, 0), (2, 0), (1, 1), (2, 1))


def _per_step_sharpe(returns: tuple[float, ...]) -> float:
    a = np.array([r for r in returns if np.isfinite(r)], dtype=float)
    if a.size < 2:
        return 0.0
    sd = float(a.std(ddof=1))
    return 0.0 if sd == 0.0 else float(a.mean() / sd)


def arima_research_verdict(
    y: np.ndarray,
    *,
    d: int = 0,
    grid: tuple[tuple[int, int], ...] = _DEFAULT_GRID,
    min_train: int = 504,
    refit_every: int = 21,
    cost_bps: float = 2.0,
) -> ARIMAVerdict:
    """Walk-forward ARIMA mean forecast across a (p,q) grid, DSR/PSR-gated.

    Picks the grid member with the best cost-adjusted sign-strategy per-step Sharpe,
    then deflates that Sharpe against the per-trial Sharpes of the WHOLE grid (the
    honest multiple-testing set). To be promotion-eligible a *mean* model must clear
    THREE bars: it actually forecasts better than its unconditional baseline
    (``mse_ratio < 1``), and its cost-adjusted directional strategy clears DSR ≥ 0.30
    and PSR ≥ 0.70. Requiring ``mse_ratio < 1`` is what stops a useless point
    forecast with a lucky, cost-free directional tilt from passing. Observational
    only — reports eligibility, promotes nothing. On daily equity returns the
    expected, documented outcome is ``passes = False`` (no edge).
    """
    import pandas as pd

    from quant.backtest.dsr import deflated_sharpe, probabilistic_sharpe

    evals = [
        walk_forward_arima_eval(
            y,
            config=ARIMAConfig(p=p, d=d, q=q),
            min_train=min_train,
            refit_every=refit_every,
            cost_bps=cost_bps,
        )
        for (p, q) in grid
    ]
    trial_sharpes = np.array([_per_step_sharpe(e.oos_strategy_returns) for e in evals], dtype=float)
    best_i = int(np.argmax(trial_sharpes)) if trial_sharpes.size else 0
    best = evals[best_i]
    series = pd.Series(best.oos_strategy_returns, dtype=float)

    if len(series) < 2:
        return ARIMAVerdict(
            best_p=best.p,
            best_q=best.q,
            n_oos=best.n_oos,
            mean_ic=best.mean_ic,
            ic_tstat=best.ic_tstat,
            hit_rate=best.hit_rate,
            mse_ratio=best.mse_ratio,
            deflated_sharpe=None,
            probabilistic_sharpe=None,
            passes_dsr=False,
            passes_psr=False,
            passes=False,
            note="insufficient OOS periods for DSR/PSR",
        )

    dsr = deflated_sharpe(series, trial_sharpes)
    psr = probabilistic_sharpe(series, 0.0)
    passes_dsr = dsr >= _DSR_GATE
    passes_psr = psr >= _PSR_GATE
    beats_baseline = best.mse_ratio is not None and best.mse_ratio < 1.0
    passes = passes_dsr and passes_psr and beats_baseline
    if passes:
        note = f"ARIMA({best.p},{d},{best.q}) is promotion-eligible (research-only)"
    elif passes_dsr and passes_psr and not beats_baseline:
        # The cost-adjusted directional tilt clears DSR/PSR but the point forecast
        # is no better than the unconditional mean — a non-edge dressed as one.
        note = (
            "directional tilt clears DSR/PSR but the point forecast does NOT beat the "
            "unconditional baseline (mse_ratio>=1) — not a real conditional-mean edge"
        )
    else:
        note = (
            "no significant conditional-mean edge — daily returns are ~unforecastable "
            "(EMH); the system forecasts variance + cross-section, not the mean"
        )
    return ARIMAVerdict(
        best_p=best.p,
        best_q=best.q,
        n_oos=best.n_oos,
        mean_ic=best.mean_ic,
        ic_tstat=best.ic_tstat,
        hit_rate=best.hit_rate,
        mse_ratio=best.mse_ratio,
        deflated_sharpe=dsr,
        probabilistic_sharpe=psr,
        passes_dsr=passes_dsr,
        passes_psr=passes_psr,
        passes=passes,
        note=note,
    )
