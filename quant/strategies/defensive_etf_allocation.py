"""Defensive ETF allocation baseline for evidence-gated paper trading."""

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
class DefensiveETFAllocation(Strategy):
    """Monthly defensive ETF allocator with SPY trend regime gate."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="defensive-etf-allocation",
        name="Defensive ETF Allocation",
        description="Top-3 blended 6/12-month ETF momentum with defensive risk-off sleeve.",
        universe=etf_universe(),
        rebalance_frequency="monthly",
        enabled_live=True,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookbacks_days": (126, 252),
        "risk_on_count": 3,
        "risk_on_cap": 0.40,
        "risk_off_assets": ("IEF", "TLT", "GLD"),
        "spy_ma_days": 200,
        "min_history_days": 252,
    }

    param_grid: ClassVar[dict[str, list[Any]]] = {
        "risk_on_count": [2, 3],
        "risk_on_cap": [0.35, 0.40],
        "spy_ma_days": [150, 200],
    }

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._close = field_frame(bars, "close")

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        return cls(bars=bars, params=params)

    def generate_signals(self, asof: date) -> pd.Series:
        loc = asof_index(pd.DatetimeIndex(self._close.index), asof)
        if loc is None or loc < int(self.params["min_history_days"]):
            return pd.Series(dtype=float)
        signals = self._blended_momentum(loc)
        if self._risk_off(loc):
            defensive = list(self.params["risk_off_assets"])
            signals = signals.reindex(defensive).dropna()
        return signals.replace([np.inf, -np.inf], np.nan).dropna()

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        loc = asof_index(pd.DatetimeIndex(self._close.index), asof)
        if loc is None or equity <= 0:
            return {}
        signals = self.generate_signals(asof)
        if signals.empty:
            return {}

        if self._risk_off(loc):
            picks = signals[signals > 0].sort_values(ascending=False)
        else:
            count = int(self.params["risk_on_count"])
            picks = signals[signals > 0].nlargest(count)
        if picks.empty:
            return {}

        weights = self._weights_for(picks.index.tolist())
        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)

    def _blended_momentum(self, loc: int) -> pd.Series:
        components: list[pd.Series] = []
        for lookback in self.params["lookbacks_days"]:
            lb = int(lookback)
            if loc < lb:
                continue
            components.append((self._close.iloc[loc] / self._close.iloc[loc - lb]) - 1.0)
        if not components:
            return pd.Series(dtype=float)
        return pd.concat(components, axis=1).mean(axis=1)

    def _risk_off(self, loc: int) -> bool:
        if "SPY" not in self._close.columns:
            return True
        ma_days = int(self.params["spy_ma_days"])
        if loc < ma_days:
            return True
        spy = self._close["SPY"].iloc[max(loc - ma_days + 1, 0) : loc + 1].dropna()
        if len(spy) < ma_days:
            return True
        return float(spy.iloc[-1]) < float(spy.mean())

    def _weights_for(self, symbols: list[str]) -> pd.Series:
        if not symbols:
            return pd.Series(dtype=float)
        if set(symbols).issubset(set(self.params["risk_off_assets"])):
            return pd.Series(1.0 / len(symbols), index=symbols)
        cap = float(self.params["risk_on_cap"])
        raw = pd.Series(1.0 / len(symbols), index=symbols)
        capped = raw.clip(upper=cap)
        remainder = 1.0 - float(capped.sum())
        uncapped = capped[capped < cap].index
        if remainder > 0 and len(uncapped) > 0:
            capped.loc[uncapped] += remainder / len(uncapped)
        return capped
