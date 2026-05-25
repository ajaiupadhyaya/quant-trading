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
from quant.strategies._common import (
    asof_index,
    drawdown_leverage_factor,
    field_frame,
    size_to_shares,
)
from quant.strategies.base import Strategy, StrategySpec


@register
class CrossSectionalMomentum(Strategy):
    """Top-decile cross-sectional momentum with 200d trend filter."""

    # ``enabled_live=False`` per validation gate (2026-05-25 final):
    # Passes 4/5 gates strongly — DSR 0.836, PSR 0.991, bootstrap lower-5% +8.79%,
    # holdout 2025→2026 +18.19%, cost-robust at 30bps. Adding Daniel-Moskowitz
    # drawdown control reduced max DD from -13.24% to -12.41% but still 1/3
    # tested regimes positive (only the 2024 bull). Long-biased cross-sectional
    # momentum is regime-fragile by construction — drawdown control reduces
    # magnitude but can't flip crash regimes positive.
    # To enable live, the strategy needs a regime overlay that goes neutral
    # or short during cross-sectional dispersion collapses (a TSMOM-style
    # signal on the strategy's own equity, or a VIX-based de-risk).
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="momentum",
        name="Cross-Sectional Momentum",
        description=(
            "JT 12-1, top decile, 200d trend filter, inverse-vol sizing, "
            "Daniel-Moskowitz drawdown control on ETF universe."
        ),
        universe=etf_universe(),
        rebalance_frequency="monthly",
        enabled_live=False,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookback_months": 12,
        "skip_months": 1,
        "top_pct": 0.30,
        "trend_filter_days": 200,
        "min_history_days": 252,
        # Sizing: per-name equal risk contribution (inverse-vol weighting).
        # When False, sizing reverts to equal-dollar across picked names.
        "vol_scale_enabled": True,
        "vol_lookback_days": 60,
        "vol_target_annual": 0.10,
        # Daniel-Moskowitz "managed momentum" — scale gross exposure down as
        # the long-only universe basket draws down. Cross-sectional momentum
        # is famously fragile to regime breaks; this is the canonical fix.
        "dd_control_enabled": True,
        "dd_lookback_days": 252,
        "dd_floor": 0.20,
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
        self._returns = self._close.pct_change(fill_method=None)

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

        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None:
            return {}

        if bool(self.params["vol_scale_enabled"]):
            weights = self._vol_scaled_weights(top.index.tolist(), loc)
        else:
            weights = pd.Series(1.0 / len(top), index=top.index)
        if weights.empty:
            return {}

        if bool(self.params["dd_control_enabled"]):
            weights = weights * drawdown_leverage_factor(
                self._returns,
                loc,
                lookback_days=int(self.params["dd_lookback_days"]),
                dd_floor=float(self.params["dd_floor"]),
            )

        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)

    def _vol_scaled_weights(self, picks: list[str], loc: int) -> pd.Series:
        """Inverse-vol weights normalized to ``vol_target_annual`` total portfolio vol.

        Each name's weight is ``(target_vol / sqrt(N)) / annualized_vol``. The
        sum of |weights| is then renormalized to <= 1 by max_leverage in the
        caller. When realized vol is zero (constant series) we fall back to
        equal-weight on the picks.
        """
        vol_lb = int(self.params["vol_lookback_days"])
        win = self._returns.loc[:, picks].iloc[max(loc - vol_lb, 0) : loc + 1]
        ann_vol = win.std(ddof=1) * float(np.sqrt(252))
        ann_vol = ann_vol.replace(0.0, np.nan)
        if ann_vol.dropna().empty:
            return pd.Series(1.0 / len(picks), index=picks)
        n = len(picks)
        per_name_vol = float(self.params["vol_target_annual"]) / float(np.sqrt(n))
        weights = per_name_vol / ann_vol
        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = float(weights.abs().sum())
        if gross > 1.0 and gross > 0:
            weights = weights / gross
        return weights
