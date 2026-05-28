"""Compose the four sizing components into a single gross-exposure scalar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.sizing.components import (
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)
from quant.sizing.models import SizingConfig, SizingDecision
from quant.strategies._common import annualize_vol

_TRADING_DAYS = 252


def compute_gross(
    returns_history: np.ndarray,
    regime_label: str | None,
    config: SizingConfig,
) -> SizingDecision:
    """Build the day's gross scalar from trailing returns + yesterday's regime label.

    ``returns_history`` must contain only returns strictly before the day being
    sized (PIT). Each component no-ops to its neutral value (1.0) when its
    toggle is off or there is too little history.
    """
    arr = np.asarray(returns_history, dtype=float)

    if config.use_vol_target:
        tail = arr[-config.vol_lookback_days :]
        realized = annualize_vol(pd.Series(tail), trading_days=_TRADING_DAYS)
        vol_scale = vol_target_scale(realized, config.target_vol, config.max_leverage)
    else:
        vol_scale = 1.0

    if config.use_kelly:
        ktail = arr[-config.kelly_lookback_days :]
        if ktail.size >= 2:
            mean_ann = float(np.mean(ktail)) * _TRADING_DAYS
            var_ann = float(np.var(ktail, ddof=1)) * _TRADING_DAYS
            kelly = fractional_kelly(mean_ann, var_ann, config.kelly_fraction, config.kelly_cap)
        else:
            kelly = 1.0
    else:
        kelly = 1.0

    if config.use_drawdown:
        dtail = arr[-config.dd_lookback_days :]
        drawdown = drawdown_throttle(dtail, config.dd_floor)
    else:
        drawdown = 1.0

    regime = regime_multiplier(regime_label, config.regime_weights) if config.use_regime else 1.0

    gross = float(max(0.0, min(config.max_leverage, vol_scale * kelly * drawdown * regime)))
    return SizingDecision(
        gross=gross, vol_scale=vol_scale, kelly=kelly, drawdown=drawdown, regime=regime
    )
