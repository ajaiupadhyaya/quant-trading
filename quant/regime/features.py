"""Point-in-time market feature matrix for the regime HMM.

Every transform uses only trailing data as of each date. Standardization is
rolling (or expanding), never full-sample — full-sample scaling would leak the
future into earlier rows, the exact look-ahead the validation gate forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from pandas.api.typing import Expanding, Rolling

from quant.data import bars, macro
from quant.regime.kalman_state import kalman_local_level


@dataclass(frozen=True)
class FeatureConfig:
    realized_vol_window: int = 21
    use_term_spread: bool = True
    standardize_window: int = 252  # rolling window; 0 => expanding
    kalman_process_var: float = 1e-4
    kalman_obs_var: float = 1e-2
    min_standardize_obs: int = 60


def _standardize(col: pd.Series, window: int, min_obs: int) -> pd.Series:
    roll: Rolling[pd.Series] | Expanding[pd.Series]
    if window > 0:
        roll = col.rolling(window=window, min_periods=min_obs)
    else:
        roll = col.expanding(min_periods=min_obs)
    mean: pd.Series = roll.mean()
    std: pd.Series = roll.std(ddof=0)
    # When std == 0 (constant window), all values equal the mean: z-score is 0.
    # Using NaN here would drop perfectly valid (if boring) constant-feature rows.
    safe_std = std.where(std > 0.0, other=np.nan)
    raw_z: pd.Series = (col - mean) / safe_std
    # Fill positions where std was 0 but mean is defined with 0.0 (z = 0).
    result: pd.Series = raw_z.where(safe_std.notna() | mean.isna(), other=0.0)
    return result


def build_feature_matrix(
    *,
    spy_close: pd.Series,
    vix: pd.Series,
    dgs10: pd.Series | None,
    dgs2: pd.Series | None,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Return a date-indexed, trailing-standardized feature frame (warmup dropped)."""
    spy_close = spy_close.sort_index().astype(float)
    log_price: pd.Series = pd.Series(np.log(spy_close.to_numpy()), index=spy_close.index)
    log_ret: pd.Series = log_price.diff()
    smoothed_ret = pd.Series(
        kalman_local_level(
            log_ret.fillna(0.0).to_numpy(),
            process_var=config.kalman_process_var,
            obs_var=config.kalman_obs_var,
        ),
        index=spy_close.index,
    )
    realized_vol: pd.Series = log_ret.rolling(config.realized_vol_window).std(ddof=0)
    _rv_clipped: pd.Series = realized_vol.replace(0.0, np.nan)
    log_vol = pd.Series(np.log(_rv_clipped.to_numpy(dtype=float)), index=spy_close.index)
    running_peak: pd.Series = spy_close.cummax()
    drawdown: pd.Series = spy_close / running_peak - 1.0

    vix_aligned: pd.Series = vix.sort_index().reindex(spy_close.index).ffill()

    raw = pd.DataFrame(
        {
            "ret": smoothed_ret,
            "vol": log_vol,
            "vix": vix_aligned,
            "drawdown": drawdown,
        }
    )
    if config.use_term_spread:
        if dgs10 is None or dgs2 is None:
            raise ValueError("use_term_spread=True requires dgs10 and dgs2 series")
        spread: pd.Series = (dgs10.sort_index().reindex(spy_close.index).ffill()) - (
            dgs2.sort_index().reindex(spy_close.index).ffill()
        )
        raw["term_spread"] = spread

    standardized: pd.DataFrame = raw.apply(
        lambda c: _standardize(c, config.standardize_window, config.min_standardize_obs)
    )
    result: pd.DataFrame = standardized.dropna()
    return result


def load_market_features(start: date, end: date, config: FeatureConfig) -> pd.DataFrame:
    """Load cached bars + FRED macro and build the feature matrix."""
    spy = bars.get_bars(bars.BarRequest(symbols=["SPY"], start=start, end=end))
    spy_close = _extract_close(spy, "SPY")
    vix = macro.get_series(macro.FRED_SERIES["vix"])
    dgs10: pd.Series | None = (
        macro.get_series(macro.FRED_SERIES["tenyear"]) if config.use_term_spread else None
    )
    dgs2: pd.Series | None = (
        macro.get_series(macro.FRED_SERIES["twoyear"]) if config.use_term_spread else None
    )
    return build_feature_matrix(spy_close=spy_close, vix=vix, dgs10=dgs10, dgs2=dgs2, config=config)


def _extract_close(frame: pd.DataFrame, symbol: str) -> pd.Series:
    """Pull the close column for `symbol` from get_bars' wide MultiIndex frame.

    ``get_bars`` always returns ``pd.concat({sym: df, ...}, axis=1)`` which
    produces MultiIndex columns ``(symbol, field)``.  A flat ``"close"`` column
    would only arise from a direct per-symbol parquet read, not from
    ``get_bars``.
    """
    if isinstance(frame.columns, pd.MultiIndex):
        close: pd.Series = frame[(symbol, "close")]
    elif "close" in frame.columns:
        close = frame["close"]
    else:
        raise KeyError(f"No close column for {symbol} in bars frame")
    return pd.Series(close, index=frame.index).astype(float)
