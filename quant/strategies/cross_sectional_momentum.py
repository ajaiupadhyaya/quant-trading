"""Cross-sectional momentum on the multi-asset ETF universe.

Jegadeesh-Titman 12-1 (lookback minus most-recent month) ranked across the
universe, top-decile long, gated by a 200-day trend filter (Faber 2007). Sized
equal-weight across selected names, monthly rebalance.

The strategy operates on the same 8 ETFs as ``trend_following`` / ``risk_parity``
to keep the bar-cache footprint small and tests fast. The cross-sectional
mechanism (rank-and-pick top-N) is what distinguishes it from time-series
trend-following.
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
class CrossSectionalMomentum(Strategy):
    """Top-decile cross-sectional momentum with 200d trend filter."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="momentum",
        name="Cross-Sectional Momentum",
        description="Jegadeesh-Titman 12-1, top decile, 200d trend filter on ETF universe.",
        universe=etf_universe(),
        rebalance_frequency="monthly",
        enabled_live=True,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookback_months": 12,
        "skip_months": 1,
        "top_pct": 0.30,
        "trend_filter_days": 200,
        "min_history_days": 252,
    }

    # Spec §2.1: lookback (6/9/12), top_pct (0.25/0.30/0.40), trend (150/200/250).
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "lookback_months": [6, 9, 12],
        "top_pct": [0.25, 0.30, 0.40],
        "trend_filter_days": [150, 200, 250],
    }

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        return cls(bars=bars, params=params)

    def generate_signals(self, asof: date) -> pd.Series:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None:
            return pd.Series(dtype=float)

        lookback = int(self.params["lookback_months"]) * 21
        skip = int(self.params["skip_months"]) * 21
        min_hist = int(self.params["min_history_days"])
        if loc < max(min_hist, lookback + skip):
            return pd.Series(dtype=float)

        end_loc = loc - skip
        start_loc = max(end_loc - lookback, 0)
        p_now = self._close.iloc[end_loc]
        p_then = self._close.iloc[start_loc]
        signal = (p_now / p_then) - 1.0

        trend_days = int(self.params["trend_filter_days"])
        window = self._close.iloc[max(loc - trend_days, 0) : loc + 1]
        ma = window.mean()
        eligible = self._close.iloc[loc] > ma
        signal = signal.where(eligible).replace([np.inf, -np.inf], np.nan).dropna()
        return signal

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        signals = self.generate_signals(asof)
        if signals.empty or equity <= 0:
            return {}
        n_pick = max(1, int(np.ceil(len(signals) * float(self.params["top_pct"]))))
        top = signals.nlargest(n_pick)
        top = top[top > 0]
        if top.empty:
            return {}
        weights = pd.Series(1.0 / len(top), index=top.index)

        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None:
            return {}
        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)
