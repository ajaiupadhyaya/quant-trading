"""Pair discovery + statistical screens for the pairs trading strategy.

Three layers, applied in order during ``discover_and_screen_pairs``:

1. **PCA candidate generation** (Avellaneda-Lee 2008) — project per-name daily
   returns onto the top ``k`` principal components, treat the residuals as
   idiosyncratic signals, then cluster by similarity in PC loading space.
   Adjacent names in the loading-space distance matrix become candidate pairs.

2. **Residual stationarity screen** (Engle-Granger flavor) — for each candidate
   ``(a, b)`` fit ``log(a) = beta log(b) + alpha + ε`` via OLS, then test whether the
   AR(1) coefficient of ε is < 1 (mean-reverting). We use a simple t-statistic
   on the AR(1) coefficient rather than pulling in statsmodels for a full ADF;
   the threshold is tuned so genuinely stationary spreads pass and random
   walks don't.

3. **Ornstein-Uhlenbeck half-life filter** — fit ``Δε_t = -θ ε_{t-1} + η`` and
   compute ``HL = ln(2) / θ``. Keep only pairs with HL in ``[min_hl, max_hl]``
   so we trade pairs that mean-revert at a useful cadence (not too fast that
   noise dominates, not too slow that capital is tied up indefinitely).

All three layers are intentionally implemented with only numpy/scipy — no
statsmodels, no scikit-learn — so the strategy stays light and the screen
behavior is fully reproducible from this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairCandidate:
    """A discovered pair with its screening statistics."""

    a: str
    b: str
    beta: float  # OLS hedge ratio: log_a = beta * log_b + alpha + eps
    alpha: float
    ar1_rho: float  # AR(1) coefficient on residuals (< 1 means mean-reverting)
    half_life_days: float  # OU half-life in trading days
    spread_std: float  # std of the residuals
    adf_stat: float = 0.0  # Engle-Granger ADF statistic (more negative = stronger rejection)
    adf_passes: bool = False  # True iff adf_stat < EG critical value at 5%


# Engle-Granger ADF critical values for residuals of a 2-variable cointegration
# regression with constant term. Values from MacKinnon (2010), Table 2.
# These differ from the standard ADF critical values because we're testing
# residuals from a pre-estimated cointegrating regression.
_EG_CV_5PCT = -3.34
_EG_CV_1PCT = -3.90


def engle_granger_adf_stat(residuals: np.ndarray, max_lag: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic on cointegration residuals.

    Implements the test from Engle & Granger (1987): regress Δε_t on ε_{t-1}
    and ``max_lag`` lagged differences, then compute the t-statistic on the
    coefficient of ε_{t-1}. Under the null of unit root the statistic follows
    the non-standard EG distribution; a value below ``_EG_CV_5PCT`` rejects
    the null (the residuals are stationary → the pair is cointegrated).

    Returns ``+inf`` on degenerate input so the downstream screen treats it
    as a non-cointegrated pair.
    """
    e = np.asarray(residuals, dtype=float)
    n = len(e)
    if n < max_lag + 5:
        return float("inf")
    de = np.diff(e)  # length n-1
    # Build regressors: ε_{t-1}, plus max_lag lagged differences.
    y = de[max_lag:]  # length n - 1 - max_lag
    n_eff = len(y)
    if n_eff < 5:
        return float("inf")
    cols: list[np.ndarray] = [e[max_lag : max_lag + n_eff]]
    for k in range(1, max_lag + 1):
        cols.append(de[max_lag - k : max_lag - k + n_eff])
    cols.append(np.ones(n_eff))  # constant
    x = np.column_stack(cols)
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    pred = x @ coef
    resid = y - pred
    sigma_sq = float((resid**2).sum() / max(n_eff - x.shape[1], 1))
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return float("inf")
    se_gamma = float(np.sqrt(sigma_sq * xtx_inv[0, 0]))
    if se_gamma <= 0.0:
        return float("inf")
    return float(coef[0] / se_gamma)


def pca_candidate_pairs(
    returns: pd.DataFrame,
    n_components: int = 5,
    max_candidates: int = 50,
) -> list[tuple[str, str]]:
    """Generate candidate pairs via PCA on the returns panel.

    Project each name's daily return series onto the top ``n_components``
    principal components — names with similar PC loadings are economically
    similar. We compute pairwise Euclidean distance in loading space and
    return the ``max_candidates`` closest unordered pairs.

    Returns an empty list if the panel is degenerate (< 2 names or all NaN).
    """
    panel = returns.dropna(axis=1, how="any")
    if panel.shape[1] < 2 or panel.shape[0] < n_components + 5:
        return []

    # Center each series; SVD gives loadings directly.
    centered = panel.values - panel.values.mean(axis=0, keepdims=True)
    # Cap components at min(n_obs, n_names, n_components).
    k = min(n_components, panel.shape[0] - 1, panel.shape[1] - 1)
    if k < 1:
        return []
    # SVD on the (T, N) matrix. Right singular vectors V (shape N x N) are the
    # per-name loadings; take top-k columns.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    loadings = vt[:k].T  # shape (N, k)

    symbols = list(panel.columns)
    diffs = loadings[:, None, :] - loadings[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)

    # Take the smallest entries of the upper triangle, in order.
    iu, ju = np.triu_indices_from(dist, k=1)
    flat = dist[iu, ju]
    order = np.argsort(flat)
    out: list[tuple[str, str]] = []
    for idx in order:
        a, b = symbols[int(iu[idx])], symbols[int(ju[idx])]
        out.append((a, b))
        if len(out) >= max_candidates:
            break
    return out


def fit_pair(prices_a: pd.Series, prices_b: pd.Series) -> PairCandidate | None:
    """Fit one candidate pair end-to-end. Returns None if the pair degenerates.

    Steps: align series, take logs, run OLS for ``beta`` and ``alpha``, compute
    residuals ``ε``, fit AR(1) to compute the mean-reversion coefficient, and
    derive the OU half-life from the AR(1) slope.
    """
    common = prices_a.index.intersection(prices_b.index)
    if len(common) < 30:
        return None
    a = prices_a.loc[common].dropna()
    b = prices_b.loc[common].dropna()
    common = a.index.intersection(b.index)
    if len(common) < 30 or (a.loc[common] <= 0).any() or (b.loc[common] <= 0).any():
        return None
    log_a = np.log(a.loc[common].values)
    log_b = np.log(b.loc[common].values)

    # OLS: log_a = beta * log_b + alpha + eps.
    n = len(log_a)
    x_mean = float(log_b.mean())
    y_mean = float(log_a.mean())
    cov_xy = float(((log_b - x_mean) * (log_a - y_mean)).sum())
    var_x = float(((log_b - x_mean) ** 2).sum())
    if var_x <= 0:
        return None
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    resid = log_a - (beta * log_b + alpha)
    spread_std = float(resid.std(ddof=1))
    if spread_std <= 1e-9:
        return None

    # AR(1) on residuals: e_t = rho * e_{t-1} + eta.
    e_prev = resid[:-1]
    e_curr = resid[1:]
    denom = float((e_prev**2).sum())
    if denom <= 0:
        return None
    rho = float((e_prev * e_curr).sum() / denom)
    # OU continuous-time speed: theta = -ln(rho); half-life = ln(2) / theta.
    if rho >= 1.0 or rho <= 0.0:
        # rho >= 1: non-stationary (or explosive). rho <= 0: not OU-like, skip.
        return None
    theta = -np.log(rho)
    if theta <= 0:
        return None
    half_life = float(np.log(2.0) / theta)
    # Bound half-life so the result is finite even when rho is very close to 1
    if not np.isfinite(half_life):
        return None

    _ = n  # retain sample size in scope for future logging
    # Engle-Granger ADF on the cointegration residuals.
    adf_stat = engle_granger_adf_stat(resid, max_lag=1)
    adf_passes = bool(np.isfinite(adf_stat) and adf_stat < _EG_CV_5PCT)
    return PairCandidate(
        a=str(a.name) if a.name is not None else "A",
        b=str(b.name) if b.name is not None else "B",
        beta=float(beta),
        alpha=float(alpha),
        ar1_rho=rho,
        half_life_days=half_life,
        spread_std=spread_std,
        adf_stat=float(adf_stat) if np.isfinite(adf_stat) else 0.0,
        adf_passes=adf_passes,
    )


def discover_and_screen_pairs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    n_components: int = 5,
    max_candidates: int = 50,
    min_half_life: float = 1.0,
    max_half_life: float = 30.0,
    min_ar1_rho: float = 0.0,
    max_ar1_rho: float = 0.95,
    max_kept: int = 20,
    require_adf: bool = True,
) -> list[PairCandidate]:
    """End-to-end discovery: PCA -> fit -> half-life + AR(1) + ADF filter.

    ``prices`` is a wide close-price frame indexed by date; ``returns`` is its
    pct-change. We return up to ``max_kept`` PairCandidate records, sorted by
    half-life ascending (faster reversion first). ``require_adf=True`` adds
    an Engle-Granger ADF stationarity gate (p < 5%) on top of the AR(1) and
    half-life screens — spec §2.3 "≥2 cointegration tests pass" lives here.
    """
    candidates = pca_candidate_pairs(
        returns, n_components=n_components, max_candidates=max_candidates
    )
    if not candidates:
        return []

    fits: list[PairCandidate] = []
    for a, b in candidates:
        if a not in prices.columns or b not in prices.columns:
            continue
        # Pass named series so the candidate carries the symbol back.
        series_a = prices[a].rename(a)
        series_b = prices[b].rename(b)
        fit = fit_pair(series_a, series_b)
        if fit is None:
            continue
        if not (min_ar1_rho < fit.ar1_rho < max_ar1_rho):
            continue
        if not (min_half_life <= fit.half_life_days <= max_half_life):
            continue
        if require_adf and not fit.adf_passes:
            continue
        fits.append(fit)

    fits.sort(key=lambda f: f.half_life_days)
    return fits[:max_kept]
