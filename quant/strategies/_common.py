"""Shared helpers for concrete strategy implementations.

Every strategy works off the same wide-format bar frame produced by
``quant.data.bars.get_bars`` — MultiIndex columns ``(symbol, field)`` and a
DatetimeIndex. These helpers cover the boring parts: pulling a price field,
resolving an as-of timestamp to the most-recent trading day, sizing whole
shares from a dollar budget.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def field_frame(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract a per-field wide frame from a (symbol, field) MultiIndex bars frame."""
    if isinstance(bars.columns, pd.MultiIndex):
        try:
            df = bars.xs(field, axis=1, level=1)
        except KeyError:
            return pd.DataFrame(index=bars.index)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df.copy()
    return bars.copy()


def asof_index(history: pd.DatetimeIndex, asof: date) -> int | None:
    """Locate ``asof`` (or the most recent earlier bar) in ``history``."""
    ts = pd.Timestamp(asof)
    if ts in history:
        loc = history.get_loc(ts)
        if not isinstance(loc, int):
            return None
        return loc
    past = history[history <= ts]
    if len(past) == 0:
        return None
    loc = history.get_loc(past[-1])
    if not isinstance(loc, int):
        return None
    return loc


def size_to_shares(
    weights: pd.Series,
    prices: pd.Series,
    equity: float,
) -> dict[str, int]:
    """Convert per-symbol portfolio weights into integer share counts.

    ``weights`` may be signed (positive = long, negative = short). Symbols with
    zero or missing price are dropped silently.
    """
    out: dict[str, int] = {}
    if equity <= 0 or weights.empty:
        return out
    for sym, w in weights.items():
        sym_str = str(sym)
        if not np.isfinite(w) or w == 0.0:
            continue
        if sym_str not in prices.index:
            continue
        price = float(prices.loc[sym_str])
        if not np.isfinite(price) or price <= 0.0:
            continue
        dollars = float(w) * equity
        shares = int(dollars / price)
        if shares != 0:
            out[sym_str] = shares
    return out


def latest_prices(close: pd.DataFrame, loc: int) -> pd.Series:
    """Row of close prices at integer location ``loc``, dropping NaNs."""
    row = close.iloc[loc]
    return row.dropna()


def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized stdev of daily returns. Returns 0 on insufficient data."""
    if len(daily_returns) < 2:
        return 0.0
    vol = float(daily_returns.std(ddof=1))
    if not np.isfinite(vol) or vol <= 0.0:
        return 0.0
    return vol * float(np.sqrt(trading_days))


def drawdown_leverage_factor(
    returns: pd.DataFrame,
    loc: int,
    *,
    lookback_days: int = 252,
    dd_floor: float = 0.20,
) -> float:
    """Daniel-Moskowitz "managed momentum" exposure attenuator.

    Returns a multiplier in ``[0, 1]`` to apply to a strategy's gross exposure
    based on the trailing drawdown of an equal-weight long-only proxy basket
    of the strategy's universe. At zero drawdown the multiplier is 1.0; at
    ``-dd_floor`` or worse it's 0.0; linear ramp in between.

    Rationale: cross-sectional equity strategies (momentum, multi-factor) and
    long-only / long-biased TSMOM all share regime fragility — they get
    crushed in sharp reversal regimes (Daniel-Moskowitz 2016 "momentum
    crashes"). The proxy basket's drawdown is a fast, model-free instrument
    for the strategy's own regime risk: when the universe as a whole is in
    a deep drawdown, halve / cut / zero our exposure rather than rely on
    the signal logic to time the reversal.

    Args:
        returns: wide DataFrame of per-symbol daily returns.
        loc: integer location of the current bar in ``returns.index``.
        lookback_days: trailing window over which to compute the drawdown.
        dd_floor: drawdown magnitude at which leverage hits zero.

    Returns 1.0 on insufficient history (warm-up) or zero dd_floor.
    """
    if dd_floor <= 0:
        return 1.0
    window = returns.iloc[max(loc - lookback_days, 0) : loc + 1]
    if window.empty:
        return 1.0
    proxy = window.mean(axis=1).fillna(0.0)
    equity = (1.0 + proxy).cumprod()
    if equity.empty:
        return 1.0
    peak = float(equity.cummax().iloc[-1])
    current = float(equity.iloc[-1])
    if peak <= 0:
        return 1.0
    dd = current / peak - 1.0  # non-positive
    if dd >= 0:
        return 1.0
    factor = 1.0 + dd / dd_floor  # linear ramp from 1.0 to 0.0
    return float(max(0.0, min(1.0, factor)))
