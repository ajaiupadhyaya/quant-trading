"""Multi-factor long/short equity portfolio on a curated mega-cap universe.

Implements a price-derivable subset of the Hou-Xue-Zhang factor zoo — the
factors that don't require point-in-time fundamentals, so the strategy is
self-contained and walk-forward-clean without an EDGAR pipeline:

- ``momentum``    : 12-1 return (Jegadeesh-Titman)
- ``low_vol``     : negative of 60d realized vol (Frazzini-Pedersen "betting against beta" cousin)
- ``reversal``    : negative of 21d return (DeBondt-Thaler short-term reversal)
- ``trend``       : (price / 200d MA) - 1 (Faber long-term trend)

Each factor is z-scored cross-sectionally per day; the composite signal is the
equal-weighted mean of z-scores. Long top quintile, short bottom quintile,
dollar-neutral by sizing each leg to equity / 2.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quant.data.universe import MEGACAP_UNIVERSE
from quant.strategies import register
from quant.strategies._common import (
    asof_index,
    drawdown_leverage_factor,
    field_frame,
    size_to_shares,
)
from quant.strategies.base import Strategy, StrategySpec

__all__ = ["MEGACAP_UNIVERSE", "MultiFactor"]


def _zscore(row: pd.Series) -> pd.Series:
    valid = row.dropna()
    if len(valid) < 3:
        return pd.Series(index=row.index, dtype=float)
    mu = float(valid.mean())
    sd = float(valid.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return pd.Series(0.0, index=row.index)
    return (row - mu) / sd


@register
class MultiFactor(Strategy):
    """Composite momentum / low-vol / reversal / trend factor portfolio."""

    # 2026-05-25 go-live: enabled live with portfolio-level RegimeOverlay
    # (SPY 200dma + VIX-spike gate). Composite long-biased cross-sectional
    # equity is regime-fragile by construction; the overlay attenuates
    # exposure in crashes but cannot flip regime-gate scores positive
    # without an active inverse sleeve. Enabled for paper-trading per the
    # 2026-05-26 go-live decision; see docs/notes/2026-05-25-go-live-decisions.md.
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="multi-factor",
        name="Multi-Factor Long/Short",
        description="Composite of momentum + low-vol + reversal + trend, top/bottom quintile L/S.",
        universe=MEGACAP_UNIVERSE,
        rebalance_frequency="monthly",
        enabled_live=True,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "momentum_lookback": 252,
        "momentum_skip": 21,
        "vol_lookback": 60,
        "reversal_lookback": 21,
        "trend_lookback": 200,
        "quintile_pct": 0.20,
        "dollar_neutral": True,
        "min_history_days": 252,
        # Hou-Xue-Zhang fundamentals integration. When True, the strategy
        # additionally pulls book-to-market, gross profitability, and
        # asset-growth (negated) from SEC EDGAR (PIT-correct) and blends them
        # in equally with the price-derived factors. When False, only the
        # price-derived factors are used (the prior default behavior).
        "use_fundamentals": True,
        # EDGAR pulls are network-bound; if the cache for any name is missing
        # at build-time we silently skip that name's fundamentals factors.
        # Daniel-Moskowitz "managed momentum" — also applies to multi-factor
        # since the composite is momentum-loaded. Scales gross exposure down
        # in deep universe-wide drawdowns.
        "dd_control_enabled": True,
        "dd_lookback_days": 252,
        "dd_floor": 0.20,
        "regime_overlay_enabled": True,
        "regime_overlay_spy_ma_days": 200,
        "regime_overlay_vix_threshold": 30.0,
    }

    # Spec §2.2: quintile size, dollar-neutral on/off, lookback per factor.
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "quintile_pct": [0.15, 0.20, 0.25],
        "dollar_neutral": [True, False],
        "vol_lookback": [30, 60, 90],
        "regime_overlay_vix_threshold": [25.0, 30.0, 35.0],
    }

    def __init__(
        self,
        bars: pd.DataFrame,
        params: dict[str, Any] | None = None,
        vix: pd.Series | None = None,
        spy_bars: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")
        self._returns = self._close.pct_change(fill_method=None)
        self._vix = vix if vix is not None else _load_vix_safe()
        self._spy_bars = spy_bars if spy_bars is not None else _load_spy_bars_safe(bars)

    @classmethod
    def build(
        cls,
        bars: pd.DataFrame,
        params: dict[str, Any] | None = None,
        vix: pd.Series | None = None,
        spy_bars: pd.DataFrame | None = None,
    ) -> Strategy:
        return cls(bars=bars, params=params, vix=vix, spy_bars=spy_bars)

    def _factor_panel(self, loc: int) -> pd.DataFrame:
        close = self._close
        rets = self._returns

        mom_lb = int(self.params["momentum_lookback"])
        mom_skip = int(self.params["momentum_skip"])
        end = loc - mom_skip
        start = max(end - mom_lb, 0)
        momentum = (close.iloc[end] / close.iloc[start]) - 1.0

        vol_lb = int(self.params["vol_lookback"])
        vol_win = rets.iloc[max(loc - vol_lb, 0) : loc + 1]
        low_vol = -vol_win.std(ddof=1)

        rev_lb = int(self.params["reversal_lookback"])
        reversal = -((close.iloc[loc] / close.iloc[max(loc - rev_lb, 0)]) - 1.0)

        trend_lb = int(self.params["trend_lookback"])
        trend_win = close.iloc[max(loc - trend_lb, 0) : loc + 1]
        trend = (close.iloc[loc] / trend_win.mean()) - 1.0

        panel = pd.DataFrame(
            {
                "momentum": momentum,
                "low_vol": low_vol,
                "reversal": reversal,
                "trend": trend,
            }
        )

        if bool(self.params["use_fundamentals"]):
            asof = pd.Timestamp(close.index[loc]).date()
            fund = self._fundamentals_panel(asof, list(close.columns), close.iloc[loc])
            if not fund.empty:
                panel = panel.join(fund, how="outer")
        return panel

    def _fundamentals_panel(
        self, asof: date, symbols: list[str], prices: pd.Series
    ) -> pd.DataFrame:
        """Build PIT fundamentals factor columns for every name in ``symbols``.

        Each symbol with missing fundamentals (cache miss, no CIK, no PIT row)
        contributes NaN for the affected factor — the cross-sectional z-score
        elsewhere handles this gracefully.
        """
        from quant.data.edgar import (
            asset_growth_yoy,
            book_to_market,
            gross_profitability,
            market_cap_asof,
        )
        from quant.util.config import Settings

        try:
            data_dir = Settings().data_dir  # type: ignore[call-arg]
        except Exception:
            return pd.DataFrame()

        # Market cap = price * PIT shares-outstanding (from SEC EDGAR's dei
        # namespace). Using raw price as a market-cap proxy is WRONG across the
        # cross-section: share counts differ by orders of magnitude (AAPL ~15B
        # vs BRK-B ~1.5B), so book/price ≠ book/market-cap in rank order. Names
        # without a PIT shares-outstanding fact contribute NaN (skipped).
        btm_vals: dict[str, float] = {}
        gp_vals: dict[str, float] = {}
        inv_vals: dict[str, float] = {}
        for sym in symbols:
            try:
                p = float(prices.get(sym, float("nan")))
            except Exception:
                continue
            if not np.isfinite(p) or p <= 0:
                continue
            mcap = market_cap_asof(sym, asof, price=p, data_dir=data_dir)
            if mcap is None or not np.isfinite(mcap) or mcap <= 0:
                continue
            btm = book_to_market(sym, asof, market_cap=mcap, data_dir=data_dir)
            if btm is not None and np.isfinite(btm):
                btm_vals[sym] = btm
            gp = gross_profitability(sym, asof, data_dir=data_dir)
            if gp is not None and np.isfinite(gp):
                gp_vals[sym] = gp
            ag = asset_growth_yoy(sym, asof, data_dir=data_dir)
            if ag is not None and np.isfinite(ag):
                inv_vals[sym] = -float(ag)  # negate: low investment = positive factor

        out = pd.DataFrame(
            {
                "book_to_market": pd.Series(btm_vals),
                "profitability": pd.Series(gp_vals),
                "investment": pd.Series(inv_vals),
            }
        )
        return out

    def generate_signals(self, asof: date) -> pd.Series:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None:
            return pd.Series(dtype=float)
        if loc < int(self.params["min_history_days"]):
            return pd.Series(dtype=float)

        panel = self._factor_panel(loc)
        z = panel.apply(_zscore, axis=0)
        composite = z.mean(axis=1)
        return composite.replace([np.inf, -np.inf], np.nan).dropna()

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        signals = self.generate_signals(asof)
        if signals.empty or equity <= 0:
            return {}
        n_each = max(1, int(np.ceil(len(signals) * float(self.params["quintile_pct"]))))
        top = signals.nlargest(n_each)
        bot = signals.nsmallest(n_each)

        if bool(self.params["dollar_neutral"]):
            long_w = pd.Series(0.5 / max(len(top), 1), index=top.index)
            short_w = pd.Series(-0.5 / max(len(bot), 1), index=bot.index)
        else:
            long_w = pd.Series(1.0 / max(len(top), 1), index=top.index)
            short_w = pd.Series(dtype=float)

        weights = pd.concat([long_w, short_w])

        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None:
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

            # RegimeOverlay expects SPY in its bars frame. Multi-factor's universe
            # is megacap names (no SPY), so we pass the separately-loaded SPY
            # frame when available; otherwise SPY component silently disables
            # and only the VIX gate applies.
            overlay_bars = self._spy_bars if self._spy_bars is not None else self._bars
            overlay = RegimeOverlay(
                bars=overlay_bars,
                vix=self._vix,
                config=RegimeOverlayConfig(
                    spy_ma_days=int(self.params["regime_overlay_spy_ma_days"]),
                    vix_threshold=float(self.params["regime_overlay_vix_threshold"]),
                ),
            )
            weights = weights * overlay.factor(asof)

        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)


def _load_vix_safe() -> pd.Series | None:
    """Load VIX from FRED cache; return None on failure."""
    try:
        from quant.data.macro import vix as _vix

        return _vix()
    except Exception:
        return None


def _load_spy_bars_safe(bars: pd.DataFrame) -> pd.DataFrame | None:
    """Load SPY bars covering the same date range as ``bars``.

    Returns a MultiIndex (symbol, field) frame containing just SPY,
    suitable for passing to RegimeOverlay. Returns None if SPY can't
    be loaded (no cache, no network) — the SPY component of the
    overlay silently disables in that case.
    """
    try:
        from quant.data.bars import BarRequest, get_bars

        if len(bars.index) == 0:
            return None
        start = bars.index.min().date()
        end = bars.index.max().date()
        return get_bars(BarRequest(symbols=["SPY"], start=start, end=end))
    except Exception:
        return None
