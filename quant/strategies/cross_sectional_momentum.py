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

    # 2026-05-25 go-live: enabled live with portfolio-level RegimeOverlay
    # (SPY 200dma + VIX-spike gate) on top of the existing Daniel-Moskowitz
    # drawdown control. Validation 2026-05-26: DSR 0.747, PSR 0.970,
    # bootstrap +2.44%, holdout +16.23% (4/5 §4 gates). Regime gate fails
    # 1/3 — a structural limit of long-biased momentum that no overlay
    # short of an active inverse sleeve can flip (cash returns 0%; the
    # gate requires strictly positive). Enabled for paper-trading per the
    # 2026-05-26 go-live decision; see docs/notes/2026-05-25-go-live-decisions.md.
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="momentum",
        name="Cross-Sectional Momentum",
        description=(
            "JT 12-1, top decile, 200d trend filter, inverse-vol sizing, "
            "Daniel-Moskowitz drawdown control on ETF universe."
        ),
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
        "regime_overlay_enabled": True,
        "regime_overlay_spy_ma_days": 200,
        "regime_overlay_vix_threshold": 30.0,
    }

    # Spec §2.1: lookback (6/9/12), top_pct (0.25/0.30/0.40), trend (150/200/250).
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "lookback_months": [6, 9, 12],
        "top_pct": [0.25, 0.30, 0.40],
        "trend_filter_days": [150, 200, 250],
        "regime_overlay_vix_threshold": [25.0, 30.0, 35.0],
    }

    def __init__(
        self,
        bars: pd.DataFrame,
        params: dict[str, Any] | None = None,
        vix: pd.Series | None = None,
    ) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")
        self._returns = self._close.pct_change(fill_method=None)
        self._vix = vix if vix is not None else _load_vix_safe()

    @classmethod
    def build(
        cls,
        bars: pd.DataFrame,
        params: dict[str, Any] | None = None,
        vix: pd.Series | None = None,
    ) -> Strategy:
        return cls(bars=bars, params=params, vix=vix)

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

        if bool(self.params.get("regime_overlay_enabled", True)):
            from quant.strategies._regime_overlay import RegimeOverlay, RegimeOverlayConfig

            overlay = RegimeOverlay(
                bars=self._bars,
                vix=self._vix,
                config=RegimeOverlayConfig(
                    spy_ma_days=int(self.params["regime_overlay_spy_ma_days"]),
                    vix_threshold=float(self.params["regime_overlay_vix_threshold"]),
                ),
            )
            weights = weights * overlay.factor(asof)

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


def _load_vix_safe() -> pd.Series | None:
    """Load VIX from the FRED cache; return None if unavailable.

    Used by the RegimeOverlay component. Walk-forward backtests run via the
    StrategyFactory signature ``(params, bars) -> Strategy`` which doesn't
    plumb VIX explicitly, so the strategy loads it itself. If the FRED cache
    hasn't been populated (e.g., in unit tests with a tmp data_dir), we
    silently degrade to no-VIX — RegimeOverlay handles ``vix=None`` by
    disabling its VIX component.
    """
    try:
        from quant.data.macro import vix as _vix

        return _vix()
    except Exception:
        return None
