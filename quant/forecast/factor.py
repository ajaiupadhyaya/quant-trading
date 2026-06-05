"""Cross-sectional equity factor model + purged out-of-sample IC (Phase 8).

A cross-sectional return model on the large-cap operating universe (the names
with both deep bar history and SEC EDGAR fundamentals). Six factors, each
oriented so a higher exposure predicts a higher return:

  price        momentum (12-1), low_vol (-60d realized vol), reversal (-21d)
  fundamental  value (book/market), quality (gross profitability TTM),
               investment (-asset growth) — all point-in-time via quant.data.edgar

Per rebalance date each factor is winsorised + z-scored cross-sectionally; the
equal-weight composite of the available z-scores is the score. A numpy ridge
variant is provided for comparison. The model is judged the only honest way:
a purged, monthly-rebalance, 21-day-forward walk-forward measuring the
cross-sectional Information Coefficient (Pearson + Spearman rank-IC) with a
t-stat — never an in-sample fit. Advisory/shadow only; any portfolio tilt is a
separate green-light (the roadmap research→promote gate). Large-cap factor
premia are modest, so the honest result may be weak — it is reported as-is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

# Large-cap operating companies with bars + EDGAR (the 57-symbol cache minus ETFs).
FACTOR_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "ABT",
    "ADBE",
    "AMZN",
    "AXP",
    "BAC",
    "BMY",
    "BRK-B",
    "C",
    "CAT",
    "CL",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVX",
    "DE",
    "DIS",
    "EOG",
    "GE",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "KO",
    "LOW",
    "MA",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "SLB",
    "TGT",
    "TSLA",
    "USB",
    "V",
    "WFC",
    "WMT",
    "XOM",
)

_FACTORS = ("momentum", "low_vol", "reversal", "value", "quality", "investment")


@dataclass(frozen=True)
class FactorConfig:
    universe: tuple[str, ...] = FACTOR_UNIVERSE
    momentum_lookback: int = 252
    momentum_skip: int = 21
    vol_lookback: int = 60
    reversal_lookback: int = 21
    forward_days: int = 21  # forward-return horizon (≈ 1 month)
    rebalance_days: int = 21  # monthly rebalance
    min_history: int = 252  # need this much history before the first rebalance
    winsor_z: float = 3.0
    ridge_lambda: float = 10.0
    min_names: int = 12  # need this many scored names to form a cross-section
    min_factors: int = 2  # a name needs this many present factors to get a score
    embargo_days: int = 21  # purge gap between ridge train and the test date


# --------------------------------------------------------------------------- #
# Cross-sectional helpers (pure)
# --------------------------------------------------------------------------- #
def winsorized_zscore(x: pd.Series, z: float = 3.0) -> pd.Series:
    """Standardise cross-sectionally then clip to ±z. NaNs pass through."""
    v = x.astype(float)
    mu = v.mean(skipna=True)
    sd = v.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return ((v - mu) / sd).clip(-z, z)


def cross_sectional_ic(scores: pd.Series, fwd: pd.Series) -> tuple[float | None, float | None]:
    """(Pearson IC, Spearman rank-IC) between scores and forward returns."""
    df = pd.concat([scores, fwd], axis=1, keys=["s", "f"]).dropna()
    if len(df) < 5:
        return None, None
    pear = float(df["s"].corr(df["f"]))
    rank = float(df["s"].corr(df["f"], method="spearman"))
    pear = pear if np.isfinite(pear) else None
    rank = rank if np.isfinite(rank) else None
    return pear, rank


def _tertile_spread(scores: pd.Series, fwd: pd.Series) -> float | None:
    """Mean forward return of the top tertile minus the bottom tertile by score."""
    df = pd.concat([scores, fwd], axis=1, keys=["s", "f"]).dropna().sort_values("s")
    n = len(df)
    if n < 6:
        return None
    k = max(1, n // 3)
    return float(df["f"].iloc[-k:].mean() - df["f"].iloc[:k].mean())


# --------------------------------------------------------------------------- #
# Factor panel construction
# --------------------------------------------------------------------------- #
def _price_factors(closes: pd.DataFrame, loc: int, cfg: FactorConfig) -> pd.DataFrame:
    """Momentum / low-vol / reversal for every symbol, using data ≤ ``loc``."""
    out: dict[str, pd.Series] = {}
    mom_end = loc - cfg.momentum_skip
    mom_start = loc - cfg.momentum_lookback
    if mom_start >= 0 and mom_end > mom_start:
        out["momentum"] = closes.iloc[mom_end] / closes.iloc[mom_start] - 1.0
    if loc - cfg.vol_lookback >= 0:
        rets = closes.iloc[loc - cfg.vol_lookback : loc + 1].pct_change()
        out["low_vol"] = -rets.std(ddof=0)
    if loc - cfg.reversal_lookback >= 0:
        out["reversal"] = -(closes.iloc[loc] / closes.iloc[loc - cfg.reversal_lookback] - 1.0)
    return pd.DataFrame(out)


def _fundamental_factors(
    asof: date, symbols: list[str], prices: pd.Series, data_dir: Any
) -> pd.DataFrame:
    """PIT value / quality / investment from EDGAR. Missing names → NaN."""
    from quant.data.edgar import (
        asset_growth_yoy,
        book_to_market,
        gross_profitability_ttm,
        market_cap_asof,
    )

    val: dict[str, float] = {}
    qual: dict[str, float] = {}
    inv: dict[str, float] = {}
    for sym in symbols:
        try:
            p = float(prices.get(sym, float("nan")))
        except Exception:
            continue
        if not np.isfinite(p) or p <= 0:
            continue
        try:
            mcap = market_cap_asof(sym, asof, price=p, data_dir=data_dir)
        except Exception:
            mcap = None
        if mcap is not None and np.isfinite(mcap) and mcap > 0:
            try:
                btm = book_to_market(sym, asof, market_cap=mcap, data_dir=data_dir)
                if btm is not None and np.isfinite(btm):
                    val[sym] = btm
            except Exception:
                pass
        try:
            gp = gross_profitability_ttm(sym, asof, data_dir=data_dir)
            if gp is not None and np.isfinite(gp):
                qual[sym] = gp
        except Exception:
            pass
        try:
            ag = asset_growth_yoy(sym, asof, data_dir=data_dir)
            if ag is not None and np.isfinite(ag):
                inv[sym] = -float(ag)  # low investment = positive factor
        except Exception:
            pass
    return pd.DataFrame(
        {"value": pd.Series(val), "quality": pd.Series(qual), "investment": pd.Series(inv)}
    )


def build_factor_panel(
    closes: pd.DataFrame, loc: int, *, data_dir: Any = None, config: FactorConfig | None = None
) -> pd.DataFrame:
    """Raw factor exposures (index = symbol, columns = factors) as of ``loc``."""
    cfg = config or FactorConfig()
    price = _price_factors(closes, loc, cfg)
    asof = pd.Timestamp(closes.index[loc]).date()
    fund = _fundamental_factors(asof, list(closes.columns), closes.iloc[loc], data_dir)
    panel = price.join(fund, how="outer")
    return panel.reindex(columns=list(_FACTORS))


def composite_score(panel: pd.DataFrame, *, config: FactorConfig | None = None) -> pd.Series:
    """Equal-weight composite of the winsorised cross-sectional factor z-scores."""
    cfg = config or FactorConfig()
    z = pd.DataFrame(
        {c: winsorized_zscore(panel[c], cfg.winsor_z) for c in panel.columns if c in _FACTORS}
    )
    present = z.notna().sum(axis=1)
    score = z.mean(axis=1, skipna=True)
    return score.where(present >= cfg.min_factors)


def _zscored_panel(panel: pd.DataFrame, cfg: FactorConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {c: winsorized_zscore(panel[c], cfg.winsor_z) for c in _FACTORS if c in panel.columns}
    )


# --------------------------------------------------------------------------- #
# Ridge (numpy closed form) — comparison model
# --------------------------------------------------------------------------- #
def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge coefficients (no intercept; inputs already standardised, y demeaned)."""
    p = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(p), x.T @ y)


# --------------------------------------------------------------------------- #
# Forward returns + rebalance schedule
# --------------------------------------------------------------------------- #
def forward_returns(closes: pd.DataFrame, loc: int, horizon: int) -> pd.Series:
    """h-day forward simple return per symbol (NaN if the window runs off the end)."""
    if loc + horizon >= len(closes):
        return pd.Series(np.nan, index=closes.columns)
    return closes.iloc[loc + horizon] / closes.iloc[loc] - 1.0


def _rebalance_locs(n: int, cfg: FactorConfig) -> list[int]:
    return list(range(cfg.min_history, n - cfg.forward_days, cfg.rebalance_days))


# --------------------------------------------------------------------------- #
# Purged walk-forward IC evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactorEval:
    model: str
    n_periods: int
    mean_ic: float | None
    ic_tstat: float | None
    ic_ir: float | None  # mean / std (per-period information ratio)
    mean_rank_ic: float | None
    rank_ic_tstat: float | None
    hit_rate: float | None  # share of periods with rank-IC > 0
    mean_tertile_spread: float | None
    per_factor_ic: dict[str, float] = field(default_factory=dict)


def _tstat(series: list[float]) -> tuple[float | None, float | None, float | None]:
    a = np.array([v for v in series if v is not None and np.isfinite(v)], dtype=float)
    if a.size < 5:
        return None, None, None
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    if sd == 0:
        return mean, None, None
    return mean, float(mean / (sd / math.sqrt(a.size))), float(mean / sd)


def walk_forward_factor_eval(
    closes: pd.DataFrame,
    *,
    data_dir: Any = None,
    config: FactorConfig | None = None,
    model: str = "composite",
) -> FactorEval:
    """Purged monthly walk-forward of cross-sectional IC. Composite needs no fit
    (every period is OOS); ridge is refit each period on past, embargo-purged data."""
    cfg = config or FactorConfig()
    closes = closes.sort_index().astype(float)
    n = len(closes)
    locs = _rebalance_locs(n, cfg)

    # Precompute each rebalance period's z-scored panel + realized forward return.
    panels: dict[int, pd.DataFrame] = {}
    fwds: dict[int, pd.Series] = {}
    for loc in locs:
        z = _zscored_panel(build_factor_panel(closes, loc, data_dir=data_dir, config=cfg), cfg)
        if z.dropna(how="all").shape[0] < cfg.min_names:
            continue
        panels[loc] = z
        fwds[loc] = forward_returns(closes, loc, cfg.forward_days)

    ics: list[float] = []
    rank_ics: list[float] = []
    spreads: list[float] = []
    factor_ics: dict[str, list[float]] = {f: [] for f in _FACTORS}
    used_locs = sorted(panels)

    for loc in used_locs:
        z = panels[loc]
        fwd = fwds[loc]
        if model == "ridge":
            score = _ridge_score_purged(loc, used_locs, panels, fwds, cfg)
        else:
            score = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= cfg.min_factors)
        if score is None:
            continue
        ic, rank = cross_sectional_ic(score, fwd)
        if ic is not None:
            ics.append(ic)
        if rank is not None:
            rank_ics.append(rank)
        sp = _tertile_spread(score, fwd)
        if sp is not None:
            spreads.append(sp)
        for f in z.columns:
            fic, _ = cross_sectional_ic(z[f], fwd)
            if fic is not None:
                factor_ics[f].append(fic)

    mean_ic, ic_t, ic_ir = _tstat(ics)
    mean_rank, rank_t, _ = _tstat(rank_ics)
    hit = float(np.mean([1.0 if r > 0 else 0.0 for r in rank_ics])) if rank_ics else None
    return FactorEval(
        model=model,
        n_periods=len(ics),
        mean_ic=mean_ic,
        ic_tstat=ic_t,
        ic_ir=ic_ir,
        mean_rank_ic=mean_rank,
        rank_ic_tstat=rank_t,
        hit_rate=hit,
        mean_tertile_spread=(float(np.mean(spreads)) if spreads else None),
        per_factor_ic={f: float(np.mean(v)) for f, v in factor_ics.items() if v},
    )


def _ridge_score_purged(
    loc: int,
    used_locs: list[int],
    panels: dict[int, pd.DataFrame],
    fwds: dict[int, pd.Series],
    cfg: FactorConfig,
) -> pd.Series | None:
    """Fit ridge on rebalances whose forward window closed ≥ embargo before ``loc``."""
    rows: list[np.ndarray] = []
    ys: list[float] = []
    cols = list(_FACTORS)
    for s in used_locs:
        if s >= loc:
            break
        if s + cfg.forward_days + cfg.embargo_days > loc:  # purge overlap
            continue
        z = panels[s].reindex(columns=cols)
        fwd = fwds[s]
        # Require only the TARGET present; a missing factor → 0 (the z-score mean),
        # consistent with the prediction side below. dropna across all factors would
        # discard every name whenever one factor (e.g. a fundamental) is absent.
        df = z.join(fwd.rename("f"))
        df = df[df["f"].notna()]
        if df.empty:
            continue
        y = df["f"].to_numpy() - df["f"].to_numpy().mean()
        rows.append(df[cols].fillna(0.0).to_numpy())
        ys.append(y)
    if len(rows) < 6:
        return None
    x = np.vstack(rows)
    y = np.concatenate(ys)
    coef = fit_ridge(x, y, cfg.ridge_lambda)
    zt = panels[loc].reindex(columns=cols)
    valid = zt.notna().sum(axis=1) >= cfg.min_factors
    zt_filled = zt.fillna(0.0)
    score = pd.Series(zt_filled.to_numpy() @ coef, index=zt.index)
    return score.where(valid)


# --------------------------------------------------------------------------- #
# Live scores (fail-open) + render
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactorScores:
    asof: str
    n_names: int
    scores: dict[str, float]  # symbol → composite score
    top: tuple[str, ...]
    bottom: tuple[str, ...]
    oos_skill: str | None


def compute_factor_scores(
    closes: pd.DataFrame,
    asof: date,
    *,
    data_dir: Any = None,
    config: FactorConfig | None = None,
    top_n: int = 5,
    oos_skill: str | None = None,
) -> FactorScores:
    """Pure: current cross-sectional composite scores + the top/bottom names."""
    cfg = config or FactorConfig()
    closes = closes.sort_index().astype(float)
    closes = closes.loc[: pd.Timestamp(asof)]
    if len(closes) < cfg.min_history:
        return FactorScores(asof.isoformat(), 0, {}, (), (), oos_skill)
    panel = build_factor_panel(closes, len(closes) - 1, data_dir=data_dir, config=cfg)
    score = composite_score(panel, config=cfg).dropna().sort_values(ascending=False)
    return FactorScores(
        asof=asof.isoformat(),
        n_names=int(score.size),
        scores={str(k): float(v) for k, v in score.items()},
        top=tuple(str(s) for s in score.index[:top_n]),
        bottom=tuple(str(s) for s in score.index[-top_n:][::-1]),
        oos_skill=oos_skill,
    )


def live_factor_scores(
    settings: Any, asof: date, *, config: FactorConfig | None = None, oos_skill: str | None = None
) -> FactorScores:
    """Bounded, fail-open factor scores from cached bars + PIT EDGAR. Never raises."""
    cfg = config or FactorConfig()
    try:
        from quant.data import bars

        data_dir = getattr(settings, "data_dir", None)
        frames: dict[str, pd.Series] = {}
        for sym in cfg.universe:
            path = bars._cache_path(sym, data_dir)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if "close" in df.columns and len(df):
                frames[sym] = df["close"]
        if not frames:
            return FactorScores(asof.isoformat(), 0, {}, (), (), oos_skill)
        closes = pd.DataFrame(frames)
        return compute_factor_scores(
            closes, asof, data_dir=data_dir, config=cfg, oos_skill=oos_skill
        )
    except Exception as exc:  # fail-open
        from quant.util.logging import logger

        logger.info("forecast.factor: live scores skipped ({!r})", exc)
        return FactorScores(asof.isoformat(), 0, {}, (), (), oos_skill)


def render_factor_scores(f: FactorScores | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if f is None or f.n_names == 0:
        return "Factor model: unavailable"
    bits = [f"{f.n_names} names"]
    if f.top:
        bits.append("long " + "/".join(f.top))
    if f.bottom:
        bits.append("short " + "/".join(f.bottom))
    if f.oos_skill:
        bits.append(f"[{f.oos_skill}]")
    return "Factor model: " + ", ".join(bits)
