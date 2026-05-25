"""Time-series momentum (TSMOM) on the 8-ETF multi-asset universe.

Moskowitz-Ooi-Pedersen 2012 ensemble: for each asset, the signal is the sign of
its past-N-month return averaged across multiple lookbacks (1/3/6/12 months).
Position sizes are vol-scaled so each asset contributes equally to portfolio
vol — target ``vol_target_annual``. Optional drawdown control scales gross
exposure down as the strategy's own equity drawdown deepens.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quant.data.universe import etf_universe
from quant.strategies import register
from quant.strategies._common import asof_index, field_frame, size_to_shares
from quant.strategies.base import Strategy, StrategySpec


@register
class TrendFollowing(Strategy):
    """Multi-lookback TSMOM ensemble with vol targeting on multi-asset ETFs."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="trend",
        name="Trend-Following ETFs",
        description="TSMOM 1/3/6/12-month ensemble, vol-targeted, long-only by default.",
        universe=etf_universe(),
        rebalance_frequency="monthly",
        enabled_live=True,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookbacks_months": (1, 3, 6, 12),
        "vol_target_annual": 0.10,
        "vol_lookback_days": 60,
        "max_leverage": 1.0,
        "allow_short": False,
        "min_history_days": 252,
    }

    # Spec §2.4: vol target (8-12%), allow_short on/off, lookback ensemble.
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "vol_target_annual": [0.08, 0.10, 0.12],
        "allow_short": [True, False],
        "lookbacks_months": [(3, 6, 12), (1, 3, 6, 12)],
    }

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")
        self._returns = self._close.pct_change(fill_method=None)

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        return cls(bars=bars, params=params)

    def _ensemble_signal(self, loc: int) -> pd.Series:
        lookbacks = [int(m) * 21 for m in self.params["lookbacks_months"]]
        components: list[pd.Series] = []
        close = self._close
        for lb in lookbacks:
            start = max(loc - lb, 0)
            if start >= loc:
                continue
            ret = (close.iloc[loc] / close.iloc[start]) - 1.0
            components.append(np.sign(ret))
        if not components:
            return pd.Series(dtype=float)
        ensemble = pd.concat(components, axis=1).mean(axis=1)
        if not bool(self.params["allow_short"]):
            ensemble = ensemble.clip(lower=0.0)
        return ensemble.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, asof: date) -> pd.Series:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]):
            return pd.Series(dtype=float)
        return self._ensemble_signal(loc)

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]) or equity <= 0:
            return {}

        signal = self._ensemble_signal(loc)
        if signal.empty or (signal == 0).all():
            return {}

        vol_lb = int(self.params["vol_lookback_days"])
        rets_win = self._returns.iloc[max(loc - vol_lb, 0) : loc + 1]
        annual_vol = rets_win.std(ddof=1) * float(np.sqrt(252))
        annual_vol = annual_vol.replace(0.0, np.nan)

        vol_target = float(self.params["vol_target_annual"])
        n_assets = max(int((signal != 0).sum()), 1)
        per_asset_vol = vol_target / float(np.sqrt(n_assets))
        weights = signal * (per_asset_vol / annual_vol)
        weights = weights.replace([np.inf, -np.inf], np.nan).dropna()

        gross = float(weights.abs().sum())
        max_lev = float(self.params["max_leverage"])
        if gross > max_lev > 0:
            weights = weights * (max_lev / gross)

        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)
