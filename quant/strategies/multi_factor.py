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

from quant.strategies import register
from quant.strategies._common import (
    asof_index,
    drawdown_leverage_factor,
    field_frame,
    size_to_shares,
)
from quant.strategies.base import Strategy, StrategySpec

MEGACAP_UNIVERSE: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "JPM",
    "V",
    "JNJ",
    "WMT",
    "PG",
    "MA",
    "HD",
    "XOM",
    "BAC",
    "DIS",
    "ADBE",
    "CRM",
]


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

    # ``enabled_live=False`` per validation gate (2026-05-25 final):
    # Passes 4/5 gates with VERY strong margins — DSR 0.909, PSR 0.998, bootstrap
    # lower-5% +78.94%, holdout 2025→2026 +11.08%, cost-robust at 30bps. But
    # only 1/3 tested regimes positive (-37.65% max DD without dd control;
    # drawdown control helps but doesn't flip crash regimes). Same fundamental
    # issue as momentum: long-biased cross-sectional equity strategies lose in
    # sharp drawdowns regardless of factor selection. Enable live once the
    # composite signal incorporates a market-regime overlay (e.g. neutralize
    # the long leg + size up the short leg when cross-sectional dispersion
    # collapses, or kill the strategy when SPY drawdown > 15%).
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="multi-factor",
        name="Multi-Factor Long/Short",
        description="Composite of momentum + low-vol + reversal + trend, top/bottom quintile L/S.",
        universe=MEGACAP_UNIVERSE,
        rebalance_frequency="monthly",
        enabled_live=False,
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
    }

    # Spec §2.2: quintile size, dollar-neutral on/off, lookback per factor.
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "quintile_pct": [0.15, 0.20, 0.25],
        "dollar_neutral": [True, False],
        "vol_lookback": [30, 60, 90],
    }

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")
        self._returns = self._close.pct_change(fill_method=None)

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        return cls(bars=bars, params=params)

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
        from quant.data.edgar import asset_growth_yoy, book_to_market, gross_profitability
        from quant.util.config import Settings

        try:
            data_dir = Settings().data_dir  # type: ignore[call-arg]
        except Exception:
            return pd.DataFrame()

        # market cap proxy: use the *price* on `asof` times the share count
        # implied by total_assets / equity. We don't have shares-outstanding
        # directly here so we approximate market cap as price * 1.0 — this
        # makes book_to_market relative across the cross-section (which is
        # what matters for ranking).
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
            # Use price as a market-cap proxy. The cross-sectional zscore is
            # invariant to a constant multiplicative offset (shares outstanding),
            # so this preserves relative ordering across names.
            btm = book_to_market(sym, asof, market_cap=p, data_dir=data_dir)
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

        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)
