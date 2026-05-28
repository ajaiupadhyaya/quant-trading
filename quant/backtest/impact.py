"""Square-root market-impact model + trailing dollar-ADV.

Pure functions taking plain values / the bars frame — no ``BacktestConfig``
import, so ``engine.py`` can import this without a circular dependency.

The impact is the size-dependent term added on top of the engine's flat
half-spread (``slippage_bps``). Undefined / degenerate inputs return 0.0 impact
(cannot estimate) rather than raising, mirroring the engine's tolerance for
sparse bars.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def market_impact_bps(
    trade_notional: float,
    adv_dollar: float,
    impact_coef_bps: float,
) -> float:
    """Square-root impact in bps: ``impact_coef_bps * sqrt(notional / adv)``.

    ``impact_coef_bps`` is the impact at 100%-of-ADV participation. Returns 0.0
    when any input is non-finite, or when ``trade_notional`` or ``adv_dollar``
    is non-positive (impact cannot be estimated).
    """
    if not (
        math.isfinite(trade_notional)
        and math.isfinite(adv_dollar)
        and math.isfinite(impact_coef_bps)
    ):
        return 0.0
    if trade_notional <= 0.0 or adv_dollar <= 0.0:
        return 0.0
    participation = trade_notional / adv_dollar
    return float(impact_coef_bps * math.sqrt(participation))


def trailing_dollar_adv(
    bars: pd.DataFrame,
    symbol: str,
    fill_ts: pd.Timestamp,
    window: int,
) -> float:
    """Mean ``close * volume`` over the ``window`` bars strictly before ``fill_ts``.

    PIT: the fill bar's own volume is excluded (only rows with index < fill_ts).
    Returns 0.0 if the (symbol, close/volume) columns are absent, there is no
    prior history, or every prior dollar-volume is non-finite.
    """
    close_col = (symbol, "close")
    vol_col = (symbol, "volume")
    if close_col not in bars.columns or vol_col not in bars.columns:
        return 0.0
    prior_index = bars.index[bars.index < fill_ts]
    if len(prior_index) == 0:
        return 0.0
    tail = prior_index[-window:]
    closes = bars[close_col].loc[tail].to_numpy(dtype=float)
    volumes = bars[vol_col].loc[tail].to_numpy(dtype=float)
    dollar_vol = closes * volumes
    dollar_vol = dollar_vol[np.isfinite(dollar_vol)]
    if len(dollar_vol) == 0:
        return 0.0
    return float(dollar_vol.mean())
