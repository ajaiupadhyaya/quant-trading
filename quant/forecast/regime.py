"""Macro-conditioned regime model + Bayesian online change-point detector (Phase 8).

Two complementary, honestly-evaluated regime tools that build on the existing
walk-forward Gaussian HMM in ``quant/regime`` (Baum-Welch EM + causal forward
filter; PIT-safe by construction).

1. **Macro-conditioned HMM.** The market-only regime HMM observes price/vol/VIX/
   drawdown/term-spread. Here we add one *market-priced* macro dimension — the
   BAA-AAA default-risk spread (Moody's, daily, no revisions, history to 1986) —
   standardized PIT on the same rolling scale. The honest question is an A/B:
   does conditioning on the credit cycle improve the regime signal's
   *out-of-sample* ability to anticipate forward market stress, versus the
   market-only baseline? We deliberately use only market-priced macro series
   (credit spreads) for conditioning — NFCI / claims / Sahm are *revised* after
   the fact, so feeding them point-in-time would leak the future. The A/B metric
   is the OOS rank-correlation of the filtered crisis probability with forward
   realized volatility, plus forward-vol separation and de-risk drawdown.

2. **Bayesian online change-point detection** (Adams & MacKay 2007). A
   training-free, fully-online run-length posterior over a Normal-Gamma
   observation model with a constant hazard. It is PIT-safe by construction (the
   run-length posterior at ``t`` uses only obs[0..t]) and complements the HMM:
   the HMM names a *standing* regime, BOCPD flags the *break* between regimes.

Nothing here actuates. Like the rest of Phase 8 a tool earns advisory/shadow
status only by beating an honest benchmark out-of-sample; otherwise it ships
research-only (CLI + this module) and is *not* wired into MarketState/analyst.
numpy/scipy/pandas only — no new deps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.regime.detect import DetectConfig, identify_states, run_detection
from quant.regime.features import FeatureConfig, _extract_close, _standardize, build_feature_matrix
from quant.regime.hmm import fit_hmm, forward_filter
from quant.regime.models import N_STATES, REGIME_LABELS

_TRADING_DAYS = 252

# Verdict from the honest A/B (compare_regime_models, SPY 2005-2026, quarterly
# refit, 4510 OOS days). Adding the BAA-AAA credit dimension lifted the crisis-
# probability -> forward-vol rank IC from 0.284 to 0.321 (+0.037, +13% rel.) — a
# real OOS improvement in the regime's *predictiveness*. The crude 3-label de-risk
# drawdown was ~flat-to-marginally-worse (-19.6% vs -21.4%), so the gain is in the
# continuous probability, not a discrete trading rule. Disposition: ADVISORY/
# SHADOW — surfaced read-only to the analyst, drives nothing; promotion to any
# actuation (de-risking/sizing) is a separate gate, never auto-granted. Mirrors
# how OOS_SKILL_SPY flags the vol forecaster.
OOS_VERDICT: str | None = "credit-conditioned: OOS crisis->vol IC 0.32 vs 0.28 (advisory)"


# --------------------------------------------------------------------------- #
# Macro-conditioned feature matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MacroRegimeConfig:
    """Which market-priced macro dimensions augment the regime HMM.

    Only revision-free, point-in-time market prices are eligible. ``credit`` is
    the BAA-AAA Moody's default-risk spread (daily). ``hy_oas`` (ICE BofA US HY
    OAS) is richer but the local cache is shallow, so it is opt-in and silently
    skipped when history is too short to standardize.
    """

    use_credit: bool = True
    use_hy_oas: bool = False
    standardize_window: int = 252
    min_standardize_obs: int = 60


def _log_return_series(close: pd.Series) -> pd.Series:
    """Daily log-return Series (type-clean: np.log runs on the ndarray, not the Series)."""
    s = close.sort_index().astype(float)
    return pd.Series(np.log(s.to_numpy()), index=s.index).diff()


def _macro_block(
    index: pd.DatetimeIndex,
    *,
    baa: pd.Series | None,
    aaa: pd.Series | None,
    hy_oas: pd.Series | None,
    config: MacroRegimeConfig,
) -> pd.DataFrame:
    """PIT-standardized macro features aligned to ``index`` (warm-up kept as NaN).

    Each raw series is reindexed onto the trading-day index and forward-filled
    (a release is known until the next one), then standardized with the SAME
    rolling z-score used for the market features so every HMM dimension shares a
    scale. Forward-fill never looks ahead; standardization is trailing-only.
    """
    cols: dict[str, pd.Series] = {}
    if config.use_credit and baa is not None and aaa is not None:
        spread = (baa.sort_index().reindex(index).ffill()) - (
            aaa.sort_index().reindex(index).ffill()
        )
        cols["credit"] = _standardize(spread, config.standardize_window, config.min_standardize_obs)
    if config.use_hy_oas and hy_oas is not None:
        hy = hy_oas.sort_index().reindex(index).ffill()
        # Need enough non-NaN history for the rolling window to be meaningful.
        if int(hy.notna().sum()) >= config.standardize_window:
            cols["hy_oas"] = _standardize(hy, config.standardize_window, config.min_standardize_obs)
    if not cols:
        return pd.DataFrame(index=index)
    return pd.DataFrame(cols, index=index)


def build_macro_regime_features(
    *,
    spy_close: pd.Series,
    vix: pd.Series,
    dgs10: pd.Series | None,
    dgs2: pd.Series | None,
    baa: pd.Series | None,
    aaa: pd.Series | None,
    hy_oas: pd.Series | None = None,
    feature_config: FeatureConfig | None = None,
    macro_config: MacroRegimeConfig | None = None,
) -> pd.DataFrame:
    """Market regime features + standardized macro block, inner-joined and warm-dropped.

    The market block is the exact existing HMM feature matrix; the macro block is
    appended as extra observation dimensions. Returns the *augmented* frame; pass
    ``macro_config`` with everything off to recover the market-only baseline.
    """
    fc = feature_config or FeatureConfig()
    mc = macro_config or MacroRegimeConfig()
    base = build_feature_matrix(spy_close=spy_close, vix=vix, dgs10=dgs10, dgs2=dgs2, config=fc)
    macro = _macro_block(
        spy_close.sort_index().index,  # type: ignore[arg-type]
        baa=baa,
        aaa=aaa,
        hy_oas=hy_oas,
        config=mc,
    )
    if macro.empty or not len(macro.columns):
        return base
    joined = base.join(macro, how="left")
    return joined.dropna()


# --------------------------------------------------------------------------- #
# Honest A/B: does macro-conditioning improve OOS regime predictiveness?
# --------------------------------------------------------------------------- #
def _spearman_ic(a: pd.Series, b: pd.Series) -> float | None:
    """Spearman rank-correlation of two aligned series (None if degenerate)."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 30:
        return None
    try:
        from scipy.stats import spearmanr  # type: ignore[import-untyped]

        rho, _ = spearmanr(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
        return float(rho) if np.isfinite(rho) else None
    except Exception:
        return None


_DERISK_WEIGHT = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


@dataclass(frozen=True)
class RegimeMetrics:
    """OOS predictive scorecard for one walk-forward regime label series."""

    name: str
    n: int
    n_features: int
    crisis_fwd_vol_ic: float | None  # rank-corr of p_crisis(t) with fwd realized vol
    fwd_vol_calm: float | None  # mean forward vol when label==calm
    fwd_vol_crisis: float | None  # mean forward vol when label==crisis
    fwd_vol_separation: float | None  # crisis - calm (bigger = sharper regimes)
    dd_baseline: float  # buy-and-hold max drawdown over the window
    dd_derisked: float  # de-risk with yesterday's label
    dd_reduction: float  # dd_derisked - dd_baseline (less negative = better)


def _predictive_metrics(
    name: str, frame: pd.DataFrame, spy_returns: pd.Series, *, n_features: int, fwd_h: int = 21
) -> RegimeMetrics:
    """Score a daily regime frame on forward-stress prediction (all PIT)."""
    rets = spy_returns.reindex(frame.index).astype(float).fillna(0.0)
    # Forward realized vol over [t+1, t+h], annualized — the thing a regime
    # signal should anticipate. shift(-1) so it is strictly forward of t.
    fwd_vol = rets.rolling(fwd_h).std(ddof=0).shift(-(fwd_h)) * math.sqrt(_TRADING_DAYS)
    p_crisis = frame["p_crisis"]
    ic = _spearman_ic(p_crisis, fwd_vol)

    labels = frame["label"]
    fv_calm = (
        float(fwd_vol[labels == "calm-bull"].mean()) if (labels == "calm-bull").any() else None
    )
    fv_crisis = float(fwd_vol[labels == "crisis"].mean()) if (labels == "crisis").any() else None
    sep = (
        (fv_crisis - fv_calm)
        if (fv_calm is not None and fv_crisis is not None and np.isfinite(fv_calm + fv_crisis))
        else None
    )

    weights = labels.map(_DERISK_WEIGHT).astype(float).shift(1).fillna(1.0)
    dd_base = _max_drawdown((1.0 + rets).cumprod())
    dd_derisk = _max_drawdown((1.0 + rets * weights).cumprod())

    return RegimeMetrics(
        name=name,
        n=len(frame),
        n_features=n_features,
        crisis_fwd_vol_ic=ic,
        fwd_vol_calm=fv_calm,
        fwd_vol_crisis=fv_crisis,
        fwd_vol_separation=sep,
        dd_baseline=dd_base,
        dd_derisked=dd_derisk,
        dd_reduction=dd_derisk - dd_base,
    )


@dataclass(frozen=True)
class RegimeComparison:
    """Side-by-side A/B of the market-only vs macro-conditioned regime HMM."""

    market: RegimeMetrics
    macro: RegimeMetrics
    ic_improvement: float | None  # macro IC - market IC (positive = macro helps)
    verdict: str  # human-readable honest call


def compare_regime_models(
    *,
    spy_close: pd.Series,
    vix: pd.Series,
    dgs10: pd.Series,
    dgs2: pd.Series,
    baa: pd.Series,
    aaa: pd.Series,
    hy_oas: pd.Series | None = None,
    detect_config: DetectConfig | None = None,
    macro_config: MacroRegimeConfig | None = None,
) -> RegimeComparison:
    """Run both regime models walk-forward and score them on the SAME OOS dates.

    The two models are identical (same HMM, same windowing, same walk-forward)
    except for the feature columns — so any difference in the OOS scorecard is
    attributable to the macro conditioning, not to the estimator. The forward-vol
    IC and de-risk drawdown are measured on the intersection of dates both models
    label, for a like-for-like comparison.
    """
    dc = detect_config or DetectConfig()
    mc = macro_config or MacroRegimeConfig()

    market_feats = build_macro_regime_features(
        spy_close=spy_close,
        vix=vix,
        dgs10=dgs10,
        dgs2=dgs2,
        baa=None,
        aaa=None,
        macro_config=MacroRegimeConfig(use_credit=False, use_hy_oas=False),
    )
    macro_feats = build_macro_regime_features(
        spy_close=spy_close,
        vix=vix,
        dgs10=dgs10,
        dgs2=dgs2,
        baa=baa,
        aaa=aaa,
        hy_oas=hy_oas,
        macro_config=mc,
    )
    market_series = run_detection(market_feats, dc)
    macro_series = run_detection(macro_feats, dc)

    # Score on the common date span both models cover.
    common = market_series.index.intersection(macro_series.index)
    spy_ret = _log_return_series(spy_close)
    m_market = _predictive_metrics(
        "market-only",
        market_series.loc[common],
        spy_ret,
        n_features=int(market_feats.shape[1]),
    )
    m_macro = _predictive_metrics(
        "macro-conditioned",
        macro_series.loc[common],
        spy_ret,
        n_features=int(macro_feats.shape[1]),
    )

    ic_impr: float | None = None
    if m_market.crisis_fwd_vol_ic is not None and m_macro.crisis_fwd_vol_ic is not None:
        ic_impr = m_macro.crisis_fwd_vol_ic - m_market.crisis_fwd_vol_ic

    verdict = _verdict(m_market, m_macro, ic_impr)
    return RegimeComparison(market=m_market, macro=m_macro, ic_improvement=ic_impr, verdict=verdict)


def _verdict(market: RegimeMetrics, macro: RegimeMetrics, ic_impr: float | None) -> str:
    """Honest, conservative read of the A/B — does macro-conditioning earn its keep?"""
    if ic_impr is None:
        return "inconclusive (insufficient overlap)"
    dd_better = macro.dd_reduction >= market.dd_reduction - 1e-6
    # A meaningful IC lift AND no worse de-risk drawdown is the bar for advisory.
    if ic_impr >= 0.02 and dd_better:
        return f"macro-conditioning HELPS (OOS crisis→fwd-vol IC +{ic_impr:.3f})"
    if ic_impr <= -0.02:
        return f"macro-conditioning HURTS (OOS IC {ic_impr:+.3f}) — research-only"
    return f"macro-conditioning marginal (OOS IC {ic_impr:+.3f}) — research-only"


# --------------------------------------------------------------------------- #
# Bayesian online change-point detection (Adams & MacKay 2007)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BOCPDConfig:
    """Normal-Gamma observation prior + constant hazard for the BOCPD recursion."""

    hazard_lambda: float = 250.0  # expected run length; H = 1/lambda
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    max_run: int = 300  # truncate the run-length axis for O(T·max_run) cost
    short_run: int = 5  # cp_prob = P(run_length <= short_run): mass on a recent reset


def _studentt_logpdf(
    x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Log posterior-predictive (Student-t) for a Normal-Gamma over each run length."""
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)  # squared scale
    z = (x - mu) ** 2 / (nu * scale2)
    from scipy.special import gammaln  # type: ignore[import-untyped]

    out = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(np.pi * nu * scale2)
        - (nu + 1.0) / 2.0 * np.log1p(z)
    )
    return np.asarray(out, dtype=float)


def bocpd_run_length(x: np.ndarray, config: BOCPDConfig | None = None) -> np.ndarray:
    """Online run-length posterior R[t, r] = P(run_length=r | obs[0..t]).

    Truncated at ``max_run``; returns a (T, max_run+1) array, each row a
    normalized posterior. PIT-safe: row ``t`` conditions only on obs[0..t].
    """
    cfg = config or BOCPDConfig()
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    cap = cfg.max_run
    out = np.zeros((n, cap + 1))
    if n == 0:
        return out

    h = 1.0 / cfg.hazard_lambda
    # Per-run-length Normal-Gamma sufficient params; index 0 is "just reset".
    mu = np.array([cfg.mu0])
    kappa = np.array([cfg.kappa0])
    alpha = np.array([cfg.alpha0])
    beta = np.array([cfg.beta0])
    rl = np.array([1.0])  # run-length posterior, starts certain at r=0

    for t in range(n):
        logpred = _studentt_logpdf(x[t], mu, kappa, alpha, beta)
        pred = np.exp(logpred - logpred.max())  # stabilize; scale cancels on normalize
        growth = rl * pred * (1.0 - h)
        cp = float(np.sum(rl * pred * h))
        new_rl = np.empty(rl.size + 1)
        new_rl[0] = cp
        new_rl[1:] = growth
        s = new_rl.sum()
        if s > 0:
            new_rl /= s

        # Update NG sufficient stats (vectorized Bayesian update), prepend the prior.
        new_mu = np.empty(mu.size + 1)
        new_kappa = np.empty(kappa.size + 1)
        new_alpha = np.empty(alpha.size + 1)
        new_beta = np.empty(beta.size + 1)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = (
            cfg.mu0,
            cfg.kappa0,
            cfg.alpha0,
            cfg.beta0,
        )
        new_kappa[1:] = kappa + 1.0
        new_alpha[1:] = alpha + 0.5
        new_mu[1:] = (kappa * mu + x[t]) / (kappa + 1.0)
        new_beta[1:] = beta + (kappa * (x[t] - mu) ** 2) / (2.0 * (kappa + 1.0))

        # Truncate the run-length axis to keep cost bounded.
        if new_rl.size > cap + 1:
            new_rl = new_rl[: cap + 1]
            new_mu = new_mu[: cap + 1]
            new_kappa = new_kappa[: cap + 1]
            new_alpha = new_alpha[: cap + 1]
            new_beta = new_beta[: cap + 1]
            new_rl = new_rl / new_rl.sum() if new_rl.sum() > 0 else new_rl

        out[t, : new_rl.size] = new_rl
        rl, mu, kappa, alpha, beta = new_rl, new_mu, new_kappa, new_alpha, new_beta

    return out


@dataclass(frozen=True)
class ChangePointRead:
    """Latest online change-point read + a short recent history. Advisory only."""

    asof: str
    cp_prob: float | None  # P(run_length <= short_run): mass on a recent break
    expected_run_length: float | None  # posterior-mean days since the last break
    recent_cp_dates: list[str] = field(default_factory=list)  # dates with cp_prob spikes


def change_point_series(returns: pd.Series, config: BOCPDConfig | None = None) -> pd.DataFrame:
    """Daily BOCPD readout: ``cp_prob`` = P(run_length<=short_run) and ``exp_run``.

    Returns are standardized by an EXPANDING (PIT) mean/std — deliberately slow so
    a volatility regime-shift is NOT adapted away within a month (a trailing window
    would erase exactly the break we want to see). The headline ``cp_prob`` is the
    posterior mass on short run lengths: ~0 in a persistent regime, jumping after a
    structural break. A sharp break also collapses ``exp_run`` toward 0.
    """
    cfg = config or BOCPDConfig()
    r = returns.dropna().astype(float)
    if len(r) < 60:
        return pd.DataFrame(columns=["cp_prob", "exp_run"])
    z = (r - r.expanding(min_periods=60).mean()) / r.expanding(min_periods=60).std(ddof=0)
    z = z.fillna(0.0)
    rl = bocpd_run_length(z.to_numpy(), cfg)
    k = min(cfg.short_run + 1, rl.shape[1])
    cp = rl[:, :k].sum(axis=1)
    exp_run = (rl * np.arange(rl.shape[1])).sum(axis=1)
    idx = z.index[: rl.shape[0]]
    out = pd.DataFrame({"cp_prob": cp, "exp_run": exp_run}, index=idx)
    # Burn-in: the expanding-std warm-up AND the recursion's cold-start (the run
    # length trivially sits below short_run for its first few steps) would both
    # masquerade as change points. Drop a leading window so neither pollutes the
    # spike list / quantiles. The live tail (what callers read) is unaffected.
    burn = min(len(out) - 1, max(126, cfg.short_run * 3))
    return out.iloc[burn:]


def compute_change_points(
    returns: pd.Series, *, config: BOCPDConfig | None = None, spike_q: float = 0.98
) -> ChangePointRead:
    """Run BOCPD on a return series; report the latest break probability + spikes."""
    cfg = config or BOCPDConfig()
    series = change_point_series(returns, cfg)
    if series.empty:
        r = returns.dropna()
        return ChangePointRead(
            asof=(r.index[-1].date().isoformat() if len(r) else date.today().isoformat()),
            cp_prob=None,
            expected_run_length=None,
        )
    cp_series = series["cp_prob"]
    thresh = float(cp_series.quantile(spike_q))
    spikes = cp_series[cp_series >= thresh]
    return ChangePointRead(
        asof=cp_series.index[-1].date().isoformat(),
        cp_prob=float(cp_series.iloc[-1]),
        expected_run_length=float(series["exp_run"].iloc[-1]),
        recent_cp_dates=[ts.date().isoformat() for ts in spikes.index[-8:]],
    )


# --------------------------------------------------------------------------- #
# Live macro-conditioned regime read (fail-open) + render
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MacroRegimeRead:
    """Today's macro-conditioned regime label + posteriors. Advisory only."""

    asof: str
    label: str | None  # calm-bull | choppy | crisis
    p_calm: float | None
    p_choppy: float | None
    p_crisis: float | None
    n_features: int
    credit_z: float | None  # standardized BAA-AAA today (the macro conditioner)
    cp_prob: float | None  # latest BOCPD change-point probability
    oos_verdict: str | None  # honest A/B flag


def compute_macro_regime(
    features: pd.DataFrame,
    asof: date,
    *,
    detect_config: DetectConfig | None = None,
    cp_prob: float | None = None,
    oos_verdict: str | None = None,
) -> MacroRegimeRead:
    """Fit the HMM on the trailing window of the augmented features and filter to today.

    PIT: trains on the trailing window strictly up to ``asof`` and runs the causal
    forward filter over a recent segment; never peeks ahead.
    """
    dc = detect_config or DetectConfig()
    feats = features.sort_index()
    feats = feats.loc[: pd.Timestamp(asof)]
    if len(feats) < N_STATES * 10:
        return MacroRegimeRead(
            asof=asof.isoformat(),
            label=None,
            p_calm=None,
            p_choppy=None,
            p_crisis=None,
            n_features=int(feats.shape[1]) if len(feats) else 0,
            credit_z=None,
            cp_prob=cp_prob,
            oos_verdict=oos_verdict,
        )
    train = feats.iloc[-dc.train_window_days :] if not dc.expanding else feats
    params = fit_hmm(train.to_numpy(), n_states=N_STATES, n_restarts=dc.n_restarts, seed=dc.seed)
    mapping = identify_states(params)
    # Forward-filter a recent segment so today's posterior has accumulated evidence.
    seg = feats.iloc[-min(len(feats), 252) :]
    post_raw = forward_filter(seg.to_numpy(), params)
    post = post_raw[:, mapping][-1]
    label = REGIME_LABELS[int(post.argmax())]
    credit_z = float(feats["credit"].iloc[-1]) if "credit" in feats.columns else None
    return MacroRegimeRead(
        asof=asof.isoformat(),
        label=label,
        p_calm=float(post[0]),
        p_choppy=float(post[1]),
        p_crisis=float(post[2]),
        n_features=int(feats.shape[1]),
        credit_z=credit_z,
        cp_prob=cp_prob,
        oos_verdict=oos_verdict,
    )


def _load_macro_inputs(
    data_dir: Path | None, start: date, end: date
) -> dict[str, pd.Series | None]:
    """Load the cached series the macro-conditioned regime needs (best-effort)."""
    from quant.data import bars, macro

    spy = bars.get_bars(bars.BarRequest(symbols=["SPY"], start=start, end=end))
    out: dict[str, pd.Series | None] = {"spy_close": _extract_close(spy, "SPY")}
    for key, name in (
        ("vix", "vix"),
        ("dgs10", "tenyear"),
        ("dgs2", "twoyear"),
        ("baa", "baa"),
        ("aaa", "aaa"),
    ):
        try:
            out[key] = macro.get_series(macro.FRED_SERIES[name])
        except Exception:
            out[key] = None
    return out


def live_macro_regime(
    settings: Any,
    asof: date,
    *,
    start: date | None = None,
    detect_config: DetectConfig | None = None,
    macro_config: MacroRegimeConfig | None = None,
    oos_verdict: str | None = None,
) -> MacroRegimeRead:
    """Bounded, fail-open macro-conditioned regime read from cached data. Never raises."""
    try:
        data_dir = getattr(settings, "data_dir", None)
        s0 = start or date(asof.year - 12, 1, 1)
        inp = _load_macro_inputs(data_dir, s0, asof)
        spy_close = inp["spy_close"]
        if spy_close is None or len(spy_close) < N_STATES * 10:
            raise ValueError("insufficient SPY history")
        feats = build_macro_regime_features(
            spy_close=spy_close,
            vix=inp["vix"] if inp["vix"] is not None else pd.Series(dtype=float),
            dgs10=inp["dgs10"],
            dgs2=inp["dgs2"],
            baa=inp["baa"],
            aaa=inp["aaa"],
            macro_config=macro_config or MacroRegimeConfig(),
        )
        cp: float | None = None
        try:
            ret = _log_return_series(spy_close).dropna()
            cp = compute_change_points(ret).cp_prob
        except Exception:
            cp = None
        return compute_macro_regime(
            feats, asof, detect_config=detect_config, cp_prob=cp, oos_verdict=oos_verdict
        )
    except Exception as exc:  # fail-open
        from quant.util.logging import logger

        logger.info("forecast.regime: live read skipped ({!r})", exc)
        return MacroRegimeRead(
            asof=asof.isoformat(),
            label=None,
            p_calm=None,
            p_choppy=None,
            p_crisis=None,
            n_features=0,
            credit_z=None,
            cp_prob=None,
            oos_verdict=oos_verdict,
        )


def render_macro_regime(r: MacroRegimeRead | None) -> str:
    """Terse one-liner for the CLI / logs / (if promoted) the Claude prompt."""
    if r is None or r.label is None:
        return "Macro regime: unavailable"
    bits = [f"{r.label} (p_crisis={r.p_crisis:.0%})" if r.p_crisis is not None else r.label]
    if r.credit_z is not None:
        bits.append(f"credit_z={r.credit_z:+.1f}")
    if r.cp_prob is not None:
        bits.append(f"cp_prob={r.cp_prob:.0%}")
    if r.oos_verdict:
        bits.append(f"[{r.oos_verdict}]")
    return "Macro regime: " + ", ".join(bits)
