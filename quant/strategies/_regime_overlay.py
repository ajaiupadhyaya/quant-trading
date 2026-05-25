"""Shared crisis de-risk overlay for regime-fragile strategies.

Cross-sectional momentum, multi-factor long/short, and pairs trading all suffer
in equity crashes (covid-2020, bear-2022) by construction. The Daniel-Moskowitz
drawdown control already in ``_common.drawdown_leverage_factor`` damps magnitude
but not enough to flip the §4 validation gate that requires ≥50% of tested
historical regimes to be positive.

This module exposes a single, well-tested implementation of crisis de-risking
that the three strategies call into rather than each inlining a near-identical
block. The overlay returns a scalar in ``[0.0, 1.0]`` that strategies multiply
into their target shares.

The factor is composed by taking the MIN (most-conservative wins) of up to
three component gates:

1. **SPY 200-day SMA breach** — when SPY's close is below its trailing 200d SMA,
   cap exposure at ``spy_halve_factor`` (default 0.5).
2. **VIX threshold** — when VIX (as-of) is at or above ``vix_threshold``, cap
   exposure at ``vix_quarter_factor`` (default 0.25).
3. **Strategy-equity 200-day SMA breach** — when the strategy's own equity
   curve is below its trailing 200d SMA, cap exposure at
   ``strategy_equity_flatten_factor`` (default 0.0 — fully flat).

All computations are point-in-time: each component uses ``asof_index`` to
resolve ``asof`` against the relevant history and never peeks at future data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant.strategies._common import asof_index, field_frame


@dataclass(frozen=True)
class RegimeOverlayConfig:
    """Parameters governing the three component gates.

    Each ``use_*_filter`` flag enables or disables the corresponding component.
    The ``*_factor`` parameters are the exposure caps applied when the
    corresponding component triggers (the overall factor is the MIN across
    triggered components, then clamped to ``[0, 1]``).
    """

    use_spy_filter: bool = True
    spy_ma_days: int = 200
    spy_halve_factor: float = 0.5

    use_vix_filter: bool = True
    vix_threshold: float = 30.0
    vix_quarter_factor: float = 0.25

    use_strategy_equity_filter: bool = False
    strategy_equity_ma_days: int = 200
    strategy_equity_flatten_factor: float = 0.0


class RegimeOverlay:
    """Stateless point-in-time crisis de-risk factor producer.

    Construct once with the data needed for all three components, then call
    :meth:`factor` for each as-of date. The returned value is meant to be
    multiplied into a strategy's target share counts.
    """

    def __init__(
        self,
        *,
        bars: pd.DataFrame,
        vix: pd.Series | None,
        config: RegimeOverlayConfig,
        strategy_equity: pd.Series | None = None,
    ) -> None:
        self._bars = bars
        self._vix = vix
        self._config = config
        self._strategy_equity = strategy_equity

    def factor(self, asof: date) -> float:
        """Return the composite exposure multiplier for ``asof`` in ``[0, 1]``."""
        cfg = self._config
        components: list[float] = [1.0]

        if cfg.use_spy_filter:
            spy_cap = self._spy_component(asof)
            if spy_cap is not None:
                components.append(spy_cap)

        if cfg.use_vix_filter:
            vix_cap = self._vix_component(asof)
            if vix_cap is not None:
                components.append(vix_cap)

        if cfg.use_strategy_equity_filter:
            eq_cap = self._strategy_equity_component(asof)
            if eq_cap is not None:
                components.append(eq_cap)

        return float(max(0.0, min(1.0, min(components))))

    # ------------------------------------------------------------------ helpers

    def _spy_component(self, asof: date) -> float | None:
        """Return the SPY-200dma cap, or ``None`` if the gate doesn't fire."""
        cfg = self._config
        bars = self._bars
        if not isinstance(bars.columns, pd.MultiIndex):
            return None
        if "SPY" not in bars.columns.get_level_values(0):
            return None
        close = field_frame(bars, "close")
        if "SPY" not in close.columns:
            return None
        history = close.index
        if not isinstance(history, pd.DatetimeIndex):
            return None
        loc = asof_index(history, asof)
        if loc is None or loc < cfg.spy_ma_days:
            return None
        window = close["SPY"].iloc[loc - cfg.spy_ma_days + 1 : loc + 1]
        sma = float(window.mean())
        current = float(close["SPY"].iloc[loc])
        if current < sma:
            return float(cfg.spy_halve_factor)
        return None

    def _vix_component(self, asof: date) -> float | None:
        """Return the VIX-threshold cap, or ``None`` if the gate doesn't fire."""
        cfg = self._config
        vix = self._vix
        if vix is None or vix.empty:
            return None
        history = vix.index
        if not isinstance(history, pd.DatetimeIndex):
            return None
        loc = asof_index(history, asof)
        if loc is None:
            return None
        current = float(vix.iloc[loc])
        if current != current:  # NaN guard
            return None
        if current >= cfg.vix_threshold:
            return float(cfg.vix_quarter_factor)
        return None

    def _strategy_equity_component(self, asof: date) -> float | None:
        """Return the strategy-equity 200dma cap, or ``None`` if it doesn't fire."""
        cfg = self._config
        equity = self._strategy_equity
        if equity is None or equity.empty:
            return None
        history = equity.index
        if not isinstance(history, pd.DatetimeIndex):
            return None
        loc = asof_index(history, asof)
        if loc is None or loc < cfg.strategy_equity_ma_days:
            return None
        window = equity.iloc[loc - cfg.strategy_equity_ma_days + 1 : loc + 1]
        sma = float(window.mean())
        current = float(equity.iloc[loc])
        if current < sma:
            return float(cfg.strategy_equity_flatten_factor)
        return None
