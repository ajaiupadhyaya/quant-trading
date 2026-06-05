"""Autonomous quant market-signals engine (advisory, read-only, NO-LLM).

A deterministic, TRAILING-ONLY battery of standard quant signals over the ETF
universe — momentum/trend, volatility term-structure, cross-asset correlation,
mean-reversion, drawdown, rates — plus a single composite risk-posture score.

This is the "autonomous quant analyst" half of the system: it computes the
quantitative situational read a human quant would form before the day, logs it
to an append-only research log, and surfaces it to the Claude decision-maker
(``quant.analyst.context``) so the two layers reason together. It performs NO
trading and changes NO governance/allocation/config — it is ADVISORY ONLY.

Design contract (mirrors ``quant.regime.features``):
  * PURE builder ``build_market_signals`` over already-loaded price/macro frames.
  * Thin, fail-open, CACHE-ONLY, time-bounded loader ``load_market_signals``.
Trailing-only is load-bearing and tested: every read truncates inputs to
``<= asof`` AND uses ``asof_index`` positional ``.iloc[loc]`` indexing, so a
future row can never leak into a past signal. Every emitted float passes
``_finite`` (NaN/inf -> None); warmup/insufficient-history -> None (never 0.0).
"""

from __future__ import annotations

import concurrent.futures
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from quant.data import bars, macro
from quant.data.universe import ETF_UNIVERSE
from quant.strategies._common import asof_index, field_frame
from quant.util.logging import logger

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalConfig:
    """Windows + thresholds. Trailing windows only; never full-sample."""

    mom_lookbacks: tuple[int, ...] = (63, 126)
    mom_skip: int = 21  # skip most-recent month (Jegadeesh-Titman 12-1)
    mom_skip_long: int = 252
    trend_ma_days: int = 200
    sma_fast: int = 50
    sma_slow: int = 200
    realized_vol_window: int = 21
    rv_long: int = 63
    rv_term_long: int = 252
    vol_of_vol_window: int = 63
    corr_window: int = 63
    disp_window: int = 21
    rsi_window: int = 14
    revert_window: int = 20
    standardize_window: int = 252
    min_standardize_obs: int = 60
    drawdown_lookback: int = 252
    annualization: int = 252
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    vix_elevated: float = 20.0
    vix_high: float = 28.0
    # Calendar days of history to request from the (cache-only) loader. Must
    # comfortably exceed the longest lookback (mom_skip+252) + std window (252).
    history_lookback_days: int = 1100
    composite_weights: tuple[tuple[str, float], ...] = (
        ("trend", 0.22),
        ("breadth", 0.10),
        ("realized_vol", 0.16),
        ("vol_term", 0.08),
        ("vrp", 0.06),
        ("avg_corr", 0.12),
        ("spy_dd", 0.14),
        ("yield_slope", 0.06),
        ("mean_rsi", 0.06),
    )
    risk_on_threshold: float = 0.33
    risk_off_threshold: float = -0.33


# Frozen/immutable, so a shared module-level default is safe (and avoids a
# function call in argument defaults — ruff B008).
_DEFAULT_CONFIG = SignalConfig()


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssetSignal:
    symbol: str
    last_close: float | None
    mom_63: float | None
    mom_126: float | None
    mom_skip_12_1: float | None
    mom_blended: float | None
    above_trend_ma: bool | None
    trend_ma_gap: float | None
    sma_cross_state: str | None  # "golden" | "death" | None
    realized_vol_ann: float | None
    vol_z: float | None
    rsi_14: float | None
    px_z_20: float | None
    drawdown: float | None


@dataclass(frozen=True)
class UniverseAggregates:
    n_assets: int
    breadth_above_trend: float | None
    median_mom_blended: float | None
    pct_positive_mom: float | None
    basket_drawdown: float | None
    max_universe_drawdown: float | None
    n_oversold: int
    n_overbought: int
    mean_rsi: float | None


@dataclass(frozen=True)
class VolBlock:
    spy_realized_vol_ann: float | None
    rv_63: float | None
    spy_vol_z: float | None
    vol_term_ratio: float | None
    vix_level: float | None
    vix_z: float | None
    vrp_proxy: float | None
    vol_of_vol: float | None
    vix_term_label: str | None
    vol_regime: str | None


@dataclass(frozen=True)
class CorrBlock:
    avg_pairwise_corr: float | None
    spy_tlt_corr: float | None
    risk_on_dispersion: float | None


@dataclass(frozen=True)
class RatesBlock:
    ust10y: float | None
    term_spread: float | None
    term_spread_z: float | None
    curve_label: str | None  # "inverted" | "normal" | None


@dataclass(frozen=True)
class MarketSignals:
    asof: date
    schema_version: int
    universe: tuple[str, ...]
    assets: tuple[AssetSignal, ...]
    aggregates: UniverseAggregates | None
    vol: VolBlock | None
    corr: CorrBlock | None
    rates: RatesBlock | None
    composite_score: float | None
    composite_label: str | None
    coverage: float | None
    n_components: int
    computable: bool = True
    degraded: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> float | None:
    """Coerce to a finite float, else None. The single sanitization gate."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _standardize(col: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Trailing rolling z-score. Verbatim port of regime.features._standardize."""
    roll = (
        col.rolling(window=window, min_periods=min_obs)
        if window > 0
        else col.expanding(min_periods=min_obs)
    )
    mean: pd.Series = roll.mean()
    std: pd.Series = roll.std(ddof=0)
    safe_std = std.where(std > 0.0, other=np.nan)
    raw_z: pd.Series = (col - mean) / safe_std
    result: pd.Series = raw_z.where(safe_std.notna() | mean.isna(), other=0.0)
    return result


def _robust_z(series: pd.Series, window: int = 252, min_obs: int = 60) -> pd.Series:
    """Trailing MAD z-score, clipped to [-4, 4].

    Median/MAD (not mean/std) so a single 2008/2020 spike does not dominate the
    normalization of the composite inputs. Constant window -> 0; warmup -> NaN.
    """
    med = series.rolling(window, min_periods=min_obs).median()
    mad = (series - med).abs().rolling(window, min_periods=min_obs).median()
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-12, np.nan)
    z = (series - med) / scale
    z = z.where(scale.notna() | med.isna(), 0.0)
    return z.clip(-4.0, 4.0)


# --------------------------------------------------------------------------- #
# Free-function signal wrappers (per-date SERIES form; unit + trailing tests
# bind to these). Every one is trailing by construction.
# --------------------------------------------------------------------------- #
def momentum(close: pd.Series, lookbacks: tuple[int, ...] = (63, 126, 252)) -> pd.DataFrame:
    """Per-lookback simple total return ``close/close.shift(lb) - 1``."""
    return pd.DataFrame({f"mom_{lb}": close / close.shift(lb) - 1.0 for lb in lookbacks})


def breadth(panel: pd.DataFrame, ma_days: int = 200) -> pd.Series:
    """Per-date fraction of names trading above their own trailing MA."""
    ma = panel.rolling(ma_days, min_periods=ma_days).mean()
    above = (panel > ma).where(ma.notna())
    return above.mean(axis=1)


def trend_filter(close: pd.Series, ma_days: int = 200) -> pd.Series:
    """Boolean: price above its trailing simple moving average."""
    ma = close.rolling(ma_days, min_periods=ma_days).mean()
    return cast(pd.Series, close > ma)


def realized_vol(close: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Trailing realized vol of log returns (ddof=0), optionally annualized."""
    lr = pd.Series(np.log(close.to_numpy(dtype=float)), index=close.index).diff()
    rv = lr.rolling(window, min_periods=window).std(ddof=0)
    return rv * math.sqrt(252) if annualize else rv


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (0-100). All-gains window -> 100; all-losses -> 0."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss > 0.0, 100.0)  # no losses in window -> RSI 100
    return out.where(avg_gain.notna(), np.nan)


def drawdown(close: pd.Series) -> pd.Series:
    """Trailing drawdown from the running (expanding) peak; values <= 0."""
    return close / close.cummax() - 1.0


def _avg_pairwise_corr_series(rets: pd.DataFrame, window: int) -> pd.Series:
    """Per-date mean of the trailing pairwise return-correlation upper triangle.

    Columns with any NaN inside the window are dropped; < 2 valid columns -> NaN.
    Trailing: the window for date t is rows ``(t-window+1 .. t)`` only.
    """
    idx = rets.index
    vals = rets.to_numpy(dtype=float)
    n, k = vals.shape
    out = np.full(n, np.nan)
    if k < 2 or n < window:
        return pd.Series(out, index=idx)
    for t in range(window - 1, n):
        w = vals[t - window + 1 : t + 1]
        colmask = ~np.isnan(w).any(axis=0)
        if int(colmask.sum()) < 2:
            continue
        wc = w[:, colmask]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(wc, rowvar=False)
        if c.ndim < 2:
            continue
        iu = np.triu_indices(c.shape[0], k=1)
        ut = c[iu]
        if ut.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            m = float(np.nanmean(ut))
        if math.isfinite(m):
            out[t] = m
    return pd.Series(out, index=idx)


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def composite_score(
    components: dict[str, float | None], config: SignalConfig
) -> tuple[float | None, str | None, float | None, int]:
    """Weight-renormalized average of present (already-squashed) component reads.

    ``components`` maps a weight key -> a value in (-1, 1) (or None when the
    component could not be computed). Missing components are dropped and the
    remaining weights renormalized — never imputed to 0. Returns
    ``(score, label, coverage, n_components)``.
    """
    weights = dict(config.composite_weights)
    present = {
        k: float(v)
        for k, v in components.items()
        if v is not None and k in weights and math.isfinite(float(v))
    }
    wsum = sum(weights[k] for k in present)
    if wsum <= 0.0:
        return (None, None, None, 0)
    score = sum(weights[k] * present[k] for k in present) / wsum
    score = max(-1.0, min(1.0, score))
    n = len(present)
    coverage = n / float(len(weights))
    if score >= config.risk_on_threshold:
        label = "risk-on"
    elif score <= config.risk_off_threshold:
        label = "risk-off"
    else:
        label = "neutral"
    return (float(score), label, float(coverage), n)


# --------------------------------------------------------------------------- #
# Per-asset + aggregate + block builders (all positional/trailing)
# --------------------------------------------------------------------------- #
def _asset_signal(
    sym: str, px: pd.Series, ret: pd.Series, loc: int, config: SignalConfig
) -> AssetSignal:
    n = len(px)

    def _mom(lb: int) -> float | None:
        j = loc - lb
        if j < 0:
            return None
        p0, p1 = float(px.iloc[j]), float(px.iloc[loc])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0.0:
            return None
        return _finite(p1 / p0 - 1.0)

    mom_63 = _mom(63)
    mom_126 = _mom(126)

    mom_skip: float | None = None
    j_end = loc - config.mom_skip
    j_start = loc - config.mom_skip - config.mom_skip_long
    if j_start >= 0:
        p0, p1 = float(px.iloc[j_start]), float(px.iloc[j_end])
        if math.isfinite(p0) and math.isfinite(p1) and p0 > 0.0:
            mom_skip = _finite(p1 / p0 - 1.0)

    avail = [m for m in (mom_63, mom_126, mom_skip) if m is not None]
    mom_blended = _finite(float(np.mean(avail))) if avail else None

    above: bool | None = None
    gap: float | None = None
    if loc - (config.trend_ma_days - 1) >= 0:
        window = px.iloc[loc - (config.trend_ma_days - 1) : loc + 1]
        if int(window.notna().sum()) >= config.trend_ma_days:
            ma = float(window.mean())
            lc = float(px.iloc[loc])
            if ma > 0.0 and math.isfinite(lc):
                gap = _finite(lc / ma - 1.0)
                above = bool(lc > ma)

    cross: str | None = None
    sma_f = px.rolling(config.sma_fast, min_periods=config.sma_fast).mean()
    sma_s = px.rolling(config.sma_slow, min_periods=config.sma_slow).mean()
    a_f, a_s = float(sma_f.iloc[loc]), float(sma_s.iloc[loc])
    if math.isfinite(a_f) and math.isfinite(a_s):
        cross = "golden" if a_f > a_s else "death"

    rv = ret.rolling(config.realized_vol_window, min_periods=config.realized_vol_window).std(ddof=0)
    rv_at = float(rv.iloc[loc])
    rv_ann = _finite(rv_at * math.sqrt(config.annualization)) if math.isfinite(rv_at) else None
    vol_z = _finite(
        _standardize(rv, config.standardize_window, config.min_standardize_obs).iloc[loc]
    )

    rsi_v = _finite(rsi(px, config.rsi_window).iloc[loc]) if n > config.rsi_window else None
    px_z = _finite(_standardize(px, config.revert_window, config.revert_window).iloc[loc])
    dd = _finite((px / px.cummax() - 1.0).iloc[loc])

    return AssetSignal(
        symbol=sym,
        last_close=_finite(px.iloc[loc]),
        mom_63=mom_63,
        mom_126=mom_126,
        mom_skip_12_1=mom_skip,
        mom_blended=mom_blended,
        above_trend_ma=above,
        trend_ma_gap=gap,
        sma_cross_state=cross,
        realized_vol_ann=rv_ann,
        vol_z=vol_z,
        rsi_14=rsi_v,
        px_z_20=px_z,
        drawdown=dd,
    )


def _aggregates(
    assets: list[AssetSignal],
    rets: pd.DataFrame,
    loc: int,
    config: SignalConfig,
) -> UniverseAggregates:
    evaluable = [a for a in assets if a.above_trend_ma is not None]
    breadth_v = (
        _finite(float(np.mean([1.0 if a.above_trend_ma else 0.0 for a in evaluable])))
        if evaluable
        else None
    )
    mom_vals = [a.mom_blended for a in assets if a.mom_blended is not None]
    median_mom = _finite(float(np.median(mom_vals))) if mom_vals else None
    pct_pos = (
        _finite(float(np.mean([1.0 if m > 0 else 0.0 for m in mom_vals]))) if mom_vals else None
    )

    basket_dd: float | None = None
    basket_ret = rets.mean(axis=1)
    start = max(0, loc - config.drawdown_lookback + 1)
    seg = basket_ret.iloc[start : loc + 1].dropna()
    if len(seg) >= 2:
        eq = (1.0 + seg).cumprod()
        basket_dd = _finite(float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0))

    dd_vals = [a.drawdown for a in assets if a.drawdown is not None]
    max_dd = _finite(min(dd_vals)) if dd_vals else None

    rsis = [a.rsi_14 for a in assets if a.rsi_14 is not None]
    n_os = sum(1 for r in rsis if r < config.rsi_oversold)
    n_ob = sum(1 for r in rsis if r > config.rsi_overbought)
    mean_rsi = _finite(float(np.mean(rsis))) if rsis else None

    return UniverseAggregates(
        n_assets=len(assets),
        breadth_above_trend=breadth_v,
        median_mom_blended=median_mom,
        pct_positive_mom=pct_pos,
        basket_drawdown=basket_dd,
        max_universe_drawdown=max_dd,
        n_oversold=n_os,
        n_overbought=n_ob,
        mean_rsi=mean_rsi,
    )


def _aligned(series: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if series is None or len(series) == 0:
        return None
    return series.sort_index().reindex(index).ffill()


def _spy_log_rv(spy: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    lr = pd.Series(np.log(spy.to_numpy(dtype=float)), index=spy.index).diff()
    rv = lr.rolling(window, min_periods=window).std(ddof=0)
    return rv * math.sqrt(252) if annualize else rv


def _vol_block(
    spy: pd.Series, vix: pd.Series | None, loc: int, config: SignalConfig
) -> VolBlock | None:
    rv21 = _spy_log_rv(spy, config.realized_vol_window)
    rv63 = _spy_log_rv(spy, config.rv_long)
    rv252 = _spy_log_rv(spy, config.rv_term_long)

    srv = _finite(rv21.iloc[loc])
    srv63 = _finite(rv63.iloc[loc])
    vol_z = _finite(
        _standardize(rv21, config.standardize_window, config.min_standardize_obs).iloc[loc]
    )

    vterm: float | None = None
    v21_at, v252_at = float(rv21.iloc[loc]), float(rv252.iloc[loc])
    if math.isfinite(v21_at) and math.isfinite(v252_at) and v252_at > 1e-12:
        vterm = _finite(v21_at / v252_at)

    vix_aligned = _aligned(vix, spy.index)
    vix_level = _finite(vix_aligned.iloc[loc]) if vix_aligned is not None else None
    vix_z = (
        _finite(
            _standardize(vix_aligned, config.standardize_window, config.min_standardize_obs).iloc[
                loc
            ]
        )
        if vix_aligned is not None
        else None
    )
    vrp = _finite(vix_level - srv * 100.0) if (vix_level is not None and srv is not None) else None
    vov = _finite(
        rv21.rolling(config.vol_of_vol_window, min_periods=config.vol_of_vol_window // 3 + 1)
        .std(ddof=0)
        .iloc[loc]
    )

    vix_term_label: str | None = None
    if vix_level is not None:
        vix_term_label = (
            "high"
            if vix_level >= config.vix_high
            else "elevated"
            if vix_level >= config.vix_elevated
            else "low"
        )
    vol_regime: str | None = None
    if vol_z is not None:
        vol_regime = "calm" if vol_z < -0.5 else "stressed" if vol_z > 1.0 else "normal"

    if all(x is None for x in (srv, srv63, vol_z, vterm, vix_level)):
        return None
    return VolBlock(
        spy_realized_vol_ann=srv,
        rv_63=srv63,
        spy_vol_z=vol_z,
        vol_term_ratio=vterm,
        vix_level=vix_level,
        vix_z=vix_z,
        vrp_proxy=vrp,
        vol_of_vol=vov,
        vix_term_label=vix_term_label,
        vol_regime=vol_regime,
    )


def _corr_block(
    closes: pd.DataFrame,
    rets: pd.DataFrame,
    loc: int,
    config: SignalConfig,
    avg_corr_series: pd.Series,
) -> CorrBlock | None:
    avg = _finite(avg_corr_series.iloc[loc]) if loc < len(avg_corr_series) else None

    spy_tlt: float | None = None
    if "SPY" in rets.columns and "TLT" in rets.columns:
        start = max(0, loc - config.corr_window + 1)
        pair = rets[["SPY", "TLT"]].iloc[start : loc + 1].dropna()
        if len(pair) >= 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                spy_tlt = _finite(pair["SPY"].corr(pair["TLT"]))

    disp: float | None = None
    r_disp = (closes / closes.shift(config.disp_window) - 1.0).iloc[loc].dropna()
    if len(r_disp) >= 2:
        disp = _finite(float(np.std(r_disp.to_numpy(dtype=float), ddof=0)))

    if avg is None and spy_tlt is None and disp is None:
        return None
    return CorrBlock(avg_pairwise_corr=avg, spy_tlt_corr=spy_tlt, risk_on_dispersion=disp)


def _rates_block(
    index: pd.Index,
    dgs10: pd.Series | None,
    dgs2: pd.Series | None,
    loc: int,
    config: SignalConfig,
) -> RatesBlock | None:
    if dgs10 is None and dgs2 is None:
        return None
    d10 = _aligned(dgs10, index)
    d2 = _aligned(dgs2, index)
    ust10y = _finite(d10.iloc[loc]) if d10 is not None else None

    ts: float | None = None
    tsz: float | None = None
    curve: str | None = None
    if d10 is not None and d2 is not None:
        spread = d10 - d2
        ts = _finite(spread.iloc[loc])
        tsz = _finite(
            _standardize(spread, config.standardize_window, config.min_standardize_obs).iloc[loc]
        )
        if ts is not None:
            curve = "inverted" if ts < 0 else "normal"

    if ust10y is None and ts is None:
        return None
    return RatesBlock(ust10y=ust10y, term_spread=ts, term_spread_z=tsz, curve_label=curve)


def _composite_components(
    closes: pd.DataFrame,
    rets: pd.DataFrame,
    spy: pd.Series | None,
    vix: pd.Series | None,
    dgs10: pd.Series | None,
    dgs2: pd.Series | None,
    avg_corr_series: pd.Series,
    loc: int,
    config: SignalConfig,
) -> dict[str, float | None]:
    """Build each composite component's trailing input series, robust-z it, apply
    the risk-on sign, and squash with tanh(z/2) into (-1, 1)."""

    def squash(series: pd.Series | None, sign: float) -> float | None:
        if series is None:
            return None
        z = _robust_z(series, config.standardize_window, config.min_standardize_obs)
        if loc >= len(z):
            return None
        zv = _finite(z.iloc[loc])
        if zv is None:
            return None
        return math.tanh(sign * zv / 2.0)

    out: dict[str, float | None] = {}
    sma200 = closes.rolling(config.sma_slow, min_periods=config.sma_slow).mean()

    trend_strength = (closes / sma200 - 1.0).clip(-0.5, 0.5).mean(axis=1)
    out["trend"] = squash(trend_strength, +1.0)

    breadth_series = (closes > sma200).where(sma200.notna()).mean(axis=1)
    out["breadth"] = squash(breadth_series, +1.0)

    rv21: pd.Series | None = None
    rv252: pd.Series | None = None
    if spy is not None:
        rv21 = _spy_log_rv(spy, config.realized_vol_window)
        rv252 = _spy_log_rv(spy, config.rv_term_long)
    out["realized_vol"] = squash(rv21, -1.0)

    vterm_series: pd.Series | None = None
    if rv21 is not None and rv252 is not None:
        denom = rv252.where(rv252 > 1e-12, np.nan)
        vterm_series = rv21 / denom
    out["vol_term"] = squash(vterm_series, -1.0)

    vix_aligned = _aligned(vix, closes.index)
    vrp_series: pd.Series | None = None
    if vix_aligned is not None and rv21 is not None:
        vrp_series = vix_aligned - rv21 * 100.0
    out["vrp"] = squash(vrp_series, +1.0)

    out["avg_corr"] = squash(avg_corr_series, -1.0)

    spy_dd_series = (spy / spy.cummax() - 1.0) if spy is not None else None
    out["spy_dd"] = squash(spy_dd_series, +1.0)

    ts_series: pd.Series | None = None
    d10 = _aligned(dgs10, closes.index)
    d2 = _aligned(dgs2, closes.index)
    if d10 is not None and d2 is not None:
        ts_series = d10 - d2
    out["yield_slope"] = squash(ts_series, +1.0)

    rsi_frame = pd.DataFrame({s: rsi(closes[s], config.rsi_window) for s in closes.columns})
    out["mean_rsi"] = squash(rsi_frame.mean(axis=1), +1.0)

    return out


# --------------------------------------------------------------------------- #
# Pure builder
# --------------------------------------------------------------------------- #
def _empty(asof: date, reason: str) -> MarketSignals:
    return MarketSignals(
        asof=asof,
        schema_version=SCHEMA_VERSION,
        universe=tuple(ETF_UNIVERSE),
        assets=(),
        aggregates=None,
        vol=None,
        corr=None,
        rates=None,
        composite_score=None,
        composite_label=None,
        coverage=None,
        n_components=0,
        computable=False,
        degraded=(reason,),
    )


def build_market_signals(
    *,
    closes: pd.DataFrame | None,
    vix: pd.Series | None,
    dgs10: pd.Series | None,
    dgs2: pd.Series | None,
    asof: date,
    config: SignalConfig = _DEFAULT_CONFIG,
) -> MarketSignals:
    """Compute the full trailing-only signal battery as of ``asof`` (pure)."""
    if closes is None or closes.empty:
        return _empty(asof, "no_bars")
    closes = closes.sort_index().astype(float)
    # Hard truncation barrier: nothing after asof can enter ANY computation.
    closes = closes.loc[: pd.Timestamp(asof)]
    closes = closes.dropna(axis=1, how="all")
    if closes.empty or closes.shape[1] == 0:
        return _empty(asof, "no_bars")

    idx = cast(pd.DatetimeIndex, closes.index)
    loc = asof_index(idx, asof)
    if loc is None:
        return _empty(asof, "no_asof")

    symbols = [str(c) for c in closes.columns]
    rets = closes.pct_change()
    spy = closes["SPY"] if "SPY" in closes.columns else None

    degraded: list[str] = []
    assets = [_asset_signal(s, closes[s], rets[s], loc, config) for s in symbols]
    aggregates = _aggregates(assets, rets, loc, config)
    avg_corr_series = _avg_pairwise_corr_series(rets, config.corr_window)

    vol = _vol_block(spy, vix, loc, config) if spy is not None else None
    if vol is None:
        degraded.append("vol")
    corr = _corr_block(closes, rets, loc, config, avg_corr_series)
    if corr is None:
        degraded.append("corr")
    rates = _rates_block(closes.index, dgs10, dgs2, loc, config)
    if rates is None:
        degraded.append("rates")

    components = _composite_components(
        closes, rets, spy, vix, dgs10, dgs2, avg_corr_series, loc, config
    )
    score, label, coverage, n_comp = composite_score(components, config)

    return MarketSignals(
        asof=asof,
        schema_version=SCHEMA_VERSION,
        universe=tuple(symbols),
        assets=tuple(assets),
        aggregates=aggregates,
        vol=vol,
        corr=corr,
        rates=rates,
        composite_score=score,
        composite_label=label,
        coverage=coverage,
        n_components=n_comp,
        computable=bool(assets),
        degraded=tuple(degraded),
    )


# --------------------------------------------------------------------------- #
# Loader (the ONE I/O entrypoint) — CACHE-ONLY + hard wall-clock budget.
# --------------------------------------------------------------------------- #
def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def _cache_bounds(asof: date, config: SignalConfig) -> tuple[date, date]:
    """The widest ``[start, end]`` window the universe cache already covers.

    ``get_bars`` fetches from the network whenever the request reaches earlier
    than ``have_start`` OR later than ``have_end``. Flooring ``end`` to the
    earliest cached end AND ceiling ``start`` to the latest cached start keeps
    every symbol's existing parquet sufficient, so the loader is cache-only. Any
    read failure -> the desired window (the in-loader timeout still bounds the
    worst case).
    """
    desired_start = asof - timedelta(days=config.history_lookback_days)
    starts: list[date] = []
    ends: list[date] = []
    for sym in ETF_UNIVERSE:
        try:
            path = bars._cache_path(sym)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if len(df):
                starts.append(df.index.min().date())
                ends.append(df.index.max().date())
        except Exception:
            continue
    if not ends:
        return desired_start, asof
    end = min(asof, min(ends))
    start = max(desired_start, max(starts))  # never earlier than the cache covers
    if start > end:
        start = end
    return start, end


def load_market_signals(
    data_dir: Path | None = None,
    asof: date | None = None,
    *,
    config: SignalConfig = _DEFAULT_CONFIG,
) -> MarketSignals:
    """Cache-only, time-bounded, fail-open load + compute. Never raises."""
    asof = asof or date.today()
    try:
        start, end = _cache_bounds(asof, config)
        req = bars.BarRequest(symbols=list(ETF_UNIVERSE), start=start, end=end)
        frame = _with_timeout(lambda: bars.get_bars(req), 20.0)
        closes = field_frame(frame, "close")
        closes = closes.loc[: pd.Timestamp(asof)]  # hard truncation barrier
    except Exception as exc:  # fail-open
        logger.info("research.signals: bars load skipped ({!r})", exc)
        return _empty(asof, "bars_error")

    def _macro(code: str) -> pd.Series | None:
        try:
            s = _with_timeout(lambda: macro.get_series(code), 10.0)
            return s if s is not None and len(s) > 0 else None
        except Exception as exc:  # one series failing must not sink the battery
            logger.info("research.signals: macro {} skipped ({!r})", code, exc)
            return None

    vix = _macro(macro.FRED_SERIES["vix"])
    dgs10 = _macro(macro.FRED_SERIES["tenyear"])
    dgs2 = _macro(macro.FRED_SERIES["twoyear"])

    try:
        return build_market_signals(
            closes=closes, vix=vix, dgs10=dgs10, dgs2=dgs2, asof=asof, config=config
        )
    except Exception as exc:  # fail-open
        logger.info("research.signals: build skipped ({!r})", exc)
        return _empty(asof, "build_error")


# --------------------------------------------------------------------------- #
# Serialization + append-only research log
# --------------------------------------------------------------------------- #
def to_json_dict(rec: MarketSignals) -> dict[str, Any]:
    return {
        "asof": rec.asof.isoformat(),
        "schema_version": rec.schema_version,
        "universe": list(rec.universe),
        "assets": [asdict(a) for a in rec.assets],
        "aggregates": asdict(rec.aggregates) if rec.aggregates else None,
        "vol": asdict(rec.vol) if rec.vol else None,
        "corr": asdict(rec.corr) if rec.corr else None,
        "rates": asdict(rec.rates) if rec.rates else None,
        "composite_score": rec.composite_score,
        "composite_label": rec.composite_label,
        "coverage": rec.coverage,
        "n_components": rec.n_components,
        "computable": rec.computable,
        "degraded": list(rec.degraded),
    }


def from_json_dict(payload: dict[str, Any]) -> MarketSignals:
    def _block(cls: Any, key: str) -> Any:
        raw = payload.get(key)
        return cls(**raw) if isinstance(raw, dict) else None

    return MarketSignals(
        asof=date.fromisoformat(str(payload["asof"])),
        schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        universe=tuple(str(s) for s in payload.get("universe", [])),
        assets=tuple(AssetSignal(**a) for a in payload.get("assets", [])),
        aggregates=_block(UniverseAggregates, "aggregates"),
        vol=_block(VolBlock, "vol"),
        corr=_block(CorrBlock, "corr"),
        rates=_block(RatesBlock, "rates"),
        composite_score=payload.get("composite_score"),
        composite_label=payload.get("composite_label"),
        coverage=payload.get("coverage"),
        n_components=int(payload.get("n_components", 0)),
        computable=bool(payload.get("computable", True)),
        degraded=tuple(str(d) for d in payload.get("degraded", [])),
    )


def signals_path(data_dir: Path) -> Path:
    return data_dir / "research" / "signals.jsonl"


def _last_logged_asof(path: Path) -> date | None:
    try:
        if not path.exists():
            return None
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                p = json.loads(line)
            except Exception:
                continue
            if isinstance(p, dict) and "asof" in p:
                return date.fromisoformat(str(p["asof"]))
    except Exception:
        return None
    return None


def append_signals(path: Path, rec: MarketSignals) -> None:
    """Idempotent per-asof, best-effort append (mirrors watch._append_decision)."""
    try:
        if _last_logged_asof(path) == rec.asof:
            return  # last-write-wins per asof; do not duplicate
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(to_json_dict(rec), sort_keys=True, allow_nan=False) + "\n")
    except Exception as exc:  # advisory log; never raise into a caller
        logger.warning("research.signals: append skipped ({!r})", exc)


def read_latest_signals(path: Path) -> MarketSignals | None:
    """Latest parseable record; blank/malformed lines skipped. Never raises."""
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return from_json_dict(payload)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Rendering (Claude prompt input + CLI + human record)
# --------------------------------------------------------------------------- #
def render_signals(rec: MarketSignals | None) -> str:
    """Terse, token-bounded one-liner. Prints the record's OWN asof so a stale
    read is visible. Unavailable/non-computable -> a clear sentinel."""
    if rec is None or not rec.computable:
        return "Research signals: unavailable"
    head = f"Research signals (asof {rec.asof.isoformat()}):"
    if rec.composite_label:
        sc = f"{rec.composite_score:+.2f}" if rec.composite_score is not None else "n/a"
        cov = f"{rec.coverage:.0%}" if rec.coverage is not None else "n/a"
        head += f" posture={rec.composite_label} score={sc} cov={cov}"
    bits: list[str] = []
    if rec.aggregates is not None:
        if rec.aggregates.breadth_above_trend is not None:
            bits.append(f"breadth={rec.aggregates.breadth_above_trend:.0%}>200d")
        if rec.aggregates.median_mom_blended is not None:
            bits.append(f"mom_med={rec.aggregates.median_mom_blended:+.1%}")
    if rec.vol is not None:
        if rec.vol.spy_realized_vol_ann is not None:
            bits.append(f"rv21={rec.vol.spy_realized_vol_ann:.0%}")
        if rec.vol.vix_level is not None:
            bits.append(f"vix={rec.vol.vix_level:.1f}")
        if rec.vol.vol_regime:
            bits.append(f"vol={rec.vol.vol_regime}")
    if rec.corr is not None and rec.corr.avg_pairwise_corr is not None:
        bits.append(f"avg_corr={rec.corr.avg_pairwise_corr:+.2f}")
    if rec.rates is not None and rec.rates.curve_label:
        ts = f"{rec.rates.term_spread:+.2f}" if rec.rates.term_spread is not None else ""
        bits.append(f"curve={rec.rates.curve_label}({ts})")
    if rec.degraded:
        bits.append(f"degraded={','.join(rec.degraded)}")
    return head + (" | " + " ".join(bits) if bits else "")
