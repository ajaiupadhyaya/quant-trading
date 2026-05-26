"""Statistical-arbitrage pairs trading with PCA discovery + cointegration screen.

Spec §2.3 SOTA pipeline:

1. **Pair discovery** — Avellaneda-Lee 2008 style: PCA on returns gives per-name
   loadings; closest pairs in loading space are candidates.
2. **Cointegration screen** — OLS hedge regression on logs, AR(1) on residuals;
   keep pairs with AR(1) coefficient strictly in (0, 0.95) (mean-reverting but
   not noise) and Ornstein-Uhlenbeck half-life in [1, 30] trading days.
3. **OLS hedge ratio** — point-in-time refit each lookback window, so the
   hedge tracks regime changes without the Kalman complexity. (A future
   iteration can swap in a Kalman filter; the entry/exit contract here doesn't
   change.)
4. **Z-score entry/exit** — enter when |z| > entry_z, exit when |z| < exit_z.
5. **Risk-parity legs** — each pair is split equal-dollar within the pair.
6. **Portfolio overlay** — equal capital across active pairs, capped at
   ``max_active_pairs``.

Discovery is gated by data availability. If the universe is too small or the
returns panel too short for PCA to fit, we fall back to ``SEED_PAIRS`` —
which keeps the strategy walk-forward-clean during the early warm-up bars.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quant.strategies import register
from quant.strategies._common import asof_index, field_frame, size_to_shares
from quant.strategies._pairs_discovery import (
    PairCandidate,
    discover_and_screen_pairs,
)
from quant.strategies.base import Strategy, StrategySpec

# Seed pairs — used as fallback when discovery is unavailable, and as the
# default trading universe. The discovery layer expands this dynamically.
SEED_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),
    ("MA", "V"),
    ("HD", "LOW"),
    ("XOM", "CVX"),
    ("WFC", "BAC"),
]

# Broader universe for PCA discovery — sector-balanced US large caps.
PAIRS_DISCOVERY_UNIVERSE: list[str] = sorted(
    {
        # Consumer staples
        "KO",
        "PEP",
        "PG",
        "WMT",
        "COST",
        "MO",
        "CL",
        # Financials
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "USB",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        # Tech
        "AAPL",
        "MSFT",
        "ORCL",
        "CSCO",
        "IBM",
        # Healthcare
        "JNJ",
        "PFE",
        "MRK",
        "ABT",
        "BMY",
        # Industrials
        "HON",
        "GE",
        "MMM",
        "CAT",
        "DE",
        # Payments / cards
        "MA",
        "V",
        "AXP",
        # Retail
        "HD",
        "LOW",
        "TGT",
    }
)

PAIRS_UNIVERSE: list[str] = PAIRS_DISCOVERY_UNIVERSE


@register
class PairsTrading(Strategy):
    """PCA-discovered pairs with cointegration screen + z-score mean reversion."""

    # 2026-05-25 re-tune (iteration 1): tighter pair selection (ADF p
    # ≤ 0.01, half-life ∈ [2, 20]), max 3 active pairs (was 5), VIX gate
    # (no trades when VIX > vix_max), and a stop-loss that forces flat at
    # |z| > stop_loss_z. Pairs alpha is structurally weak post-2010
    # (Gatev-Goetzmann-Rouwenhorst 2006 returns don't reproduce on modern
    # data), so we run only the highest-confidence pairs in low-vol
    # regimes. ``enabled_live`` flipped in Phase 3 once validation passes.
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="pairs",
        name="Pairs Trading",
        description=(
            "PCA-discovered pairs + AR(1)/half-life cointegration screen + "
            "OLS hedge + z-score entry/exit on a sector-balanced US large-cap universe."
        ),
        universe=PAIRS_UNIVERSE,
        rebalance_frequency="weekly",
        enabled_live=False,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookback_days": 60,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_active_pairs": 3,  # was 5 — concentrate on highest-confidence
        "min_history_days": 252,
        "discovery_window_days": 252,
        "min_half_life": 2.0,  # was 1.0 — drop fastest mean-reverters (noise)
        "max_half_life": 20.0,  # was 30.0 — drop slowest (likely cointegration breakdown)
        "adf_p_max": 0.01,  # ADF p-value cutoff (existing `require_adf` boolean stays)
        "stop_loss_z": 4.5,  # exit hard at |z| > 4.5
        "vix_max": 25.0,  # skip rebalance entirely when VIX > 25
        "n_pca_components": 5,
        "max_pca_candidates": 60,
        "max_screened_pairs": 20,
        "rediscover_every_days": 60,
        "require_adf": True,  # Engle-Granger ADF screen during discovery
        # Hedge regression: "ols" (rolling per window) or "kalman" (Elliott et al. 2005).
        "hedge_mode": "ols",
        "kalman_delta": 1e-5,
        "kalman_obs_var": 1e-3,
    }

    param_grid: ClassVar[dict[str, list[Any]]] = {
        "entry_z": [2.0, 2.5, 3.0],
        "exit_z": [0.0, 0.25, 0.5],
        "lookback_days": [30, 45, 60, 90],
        "stop_loss_z": [3.5, 4.5, 6.0],
        "vix_max": [20.0, 25.0, 30.0],
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
        # State held across rebalance days.
        self._state: dict[tuple[str, str], int] = {}
        self._discovered: list[PairCandidate] = []
        self._last_discovery_loc: int = -(10**9)

    @classmethod
    def build(
        cls,
        bars: pd.DataFrame,
        params: dict[str, Any] | None = None,
        vix: pd.Series | None = None,
    ) -> Strategy:
        return cls(bars=bars, params=params, vix=vix)

    # --- discovery ---------------------------------------------------------

    def _maybe_rediscover(self, loc: int) -> None:
        """Refresh ``self._discovered`` if enough bars have elapsed."""
        cadence = int(self.params["rediscover_every_days"])
        if loc - self._last_discovery_loc < cadence and self._discovered:
            return

        window = int(self.params["discovery_window_days"])
        start_loc = max(loc - window, 0)
        prices_window = self._close.iloc[start_loc : loc + 1]
        returns_window = self._returns.iloc[start_loc : loc + 1]

        # TODO: ``discover_and_screen_pairs`` does not currently accept an
        # ``adf_p_max`` kwarg — it uses a hardcoded Engle-Granger 5% critical
        # value. Plumbing a per-strategy ADF p-value cutoff (default 0.01,
        # tighter than current 5%) is a separate iteration to keep
        # ``_pairs_discovery.py`` untouched in this task.
        pairs = discover_and_screen_pairs(
            prices=prices_window,
            returns=returns_window,
            n_components=int(self.params["n_pca_components"]),
            max_candidates=int(self.params["max_pca_candidates"]),
            min_half_life=float(self.params["min_half_life"]),
            max_half_life=float(self.params["max_half_life"]),
            max_kept=int(self.params["max_screened_pairs"]),
            require_adf=bool(self.params["require_adf"]),
        )

        if not pairs:
            # Fallback to the seed list when discovery yields nothing — keeps
            # the strategy functional in tiny / synthetic universes.
            pairs = [
                fit for fit in (self._fit_seed(loc, a, b) for a, b in SEED_PAIRS) if fit is not None
            ]

        self._discovered = pairs
        self._last_discovery_loc = loc

    def _fit_seed(self, loc: int, a: str, b: str) -> PairCandidate | None:
        if a not in self._close.columns or b not in self._close.columns:
            return None
        lookback = int(self.params["discovery_window_days"])
        start = max(loc - lookback, 0)
        from quant.strategies._pairs_discovery import fit_pair

        return fit_pair(
            self._close[a].iloc[start : loc + 1].rename(a),
            self._close[b].iloc[start : loc + 1].rename(b),
        )

    # --- per-pair spread + z-score ----------------------------------------

    def _spread_z(self, pair: PairCandidate, loc: int) -> float:
        lookback = int(self.params["lookback_days"])
        start = max(loc - lookback, 0)
        a = self._close[pair.a].iloc[start : loc + 1].dropna()
        b = self._close[pair.b].iloc[start : loc + 1].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 10:
            return float("nan")
        log_a = np.log(a.loc[common].values)
        log_b = np.log(b.loc[common].values)

        if self.params.get("hedge_mode") == "kalman":
            from quant.strategies._kalman import kalman_hedge

            fit = kalman_hedge(
                log_a,
                log_b,
                delta=float(self.params["kalman_delta"]),
                obs_var=float(self.params["kalman_obs_var"]),
            )
            if fit is None:
                return float("nan")
            spread_now = log_a[-1] - fit.beta * log_b[-1] - fit.alpha
            mu = float(np.mean(fit.residuals))
            return float((spread_now - mu) / fit.spread_std)

        spread = log_a - pair.beta * log_b - pair.alpha
        mu = float(np.mean(spread))
        sd = float(np.std(spread, ddof=1))
        if sd <= 0 or not np.isfinite(sd):
            return float("nan")
        return float((spread[-1] - mu) / sd)

    # --- public API --------------------------------------------------------

    def generate_signals(self, asof: date) -> pd.Series:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]):
            return pd.Series(dtype=float)
        self._maybe_rediscover(loc)
        signals: dict[str, float] = {}
        for pair in self._discovered:
            z = self._spread_z(pair, loc)
            if np.isfinite(z):
                signals[f"{pair.a}/{pair.b}"] = z
        return pd.Series(signals)

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        # VIX gate: pairs trading is a mean-reversion play; in high-vol regimes
        # the spreads tend to widen further before reverting, blowing past
        # any reasonable z-score threshold. Skip entirely when VIX is hot.
        if self._vix is not None:
            vix_max = float(self.params.get("vix_max", 25.0))
            ts = pd.Timestamp(asof)
            recent = self._vix[self._vix.index <= ts]
            if len(recent) > 0:
                latest = float(recent.iloc[-1])
                if np.isfinite(latest) and latest > vix_max:
                    return {}

        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]) or equity <= 0:
            return {}
        self._maybe_rediscover(loc)

        entry_z = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        max_active = int(self.params["max_active_pairs"])
        stop_z = float(self.params.get("stop_loss_z", 4.5))

        decisions: list[tuple[PairCandidate, int]] = []
        for pair in self._discovered:
            key = (pair.a, pair.b)
            z = self._spread_z(pair, loc)
            if not np.isfinite(z):
                self._state[key] = 0
                continue
            if np.isfinite(z) and abs(z) > stop_z:
                # Adverse blow-out beyond stop_loss_z — force flat regardless of
                # current state. Pairs that move 4.5+ standard deviations against
                # us suggest the spread isn't mean-reverting (cointegration
                # breakdown); cut the loss.
                self._state[key] = 0
                decisions.append((pair, 0))
                continue
            current = self._state.get(key, 0)
            new_side = current
            if current == 0:
                if z > entry_z:
                    new_side = -1  # short the spread: sell A, buy hedge*B
                elif z < -entry_z:
                    new_side = +1
            elif abs(z) < exit_z:
                new_side = 0
            self._state[key] = new_side
            if new_side != 0:
                decisions.append((pair, new_side))

        if not decisions:
            return {}
        decisions = decisions[:max_active]

        per_pair_dollars = equity / len(decisions)
        weights: dict[str, float] = {}
        for pair, side in decisions:
            half = (per_pair_dollars / 2.0) / equity
            sign_b = float(np.sign(pair.beta) or 1.0)
            weights[pair.a] = weights.get(pair.a, 0.0) + side * half
            weights[pair.b] = weights.get(pair.b, 0.0) + (-side) * half * sign_b

        prices = self._close.iloc[loc].dropna()
        return size_to_shares(pd.Series(weights), prices, equity)


def _load_vix_safe() -> pd.Series | None:
    """Load VIX from FRED cache; return None on failure. See momentum strategy for usage notes."""
    try:
        from quant.data.macro import vix as _vix

        return _vix()
    except Exception:
        return None
