"""Statistical-arbitrage pairs trading on a small hand-picked set.

Each pair holds two correlated names — the strategy estimates a rolling OLS
hedge ratio on log prices, builds a spread, normalizes to a z-score, and
enters when the spread is more than ``entry_z`` standard deviations from zero,
exits when it crosses ``exit_z``. Capital is split equally across active
pairs; each leg of an active pair gets half of the pair's allocation, with
opposite signs.

The hand-picked seed pairs are well-known historical co-movers (same sector,
similar business). A future iteration would discover pairs via
PCA-on-returns + cointegration screening; until that lands, these defaults
give a working strategy that runs through the engine end-to-end.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quant.strategies import register
from quant.strategies._common import asof_index, field_frame, size_to_shares
from quant.strategies.base import Strategy, StrategySpec

SEED_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),
    ("MA", "V"),
    ("HD", "LOW"),
    ("XOM", "CVX"),
    ("WFC", "BAC"),
]

PAIRS_UNIVERSE: list[str] = sorted({s for pair in SEED_PAIRS for s in pair})


def _hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    """OLS regression of log_a on log_b through the origin (after demean)."""
    a = log_a - log_a.mean()
    b = log_b - log_b.mean()
    denom = float((b * b).sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float((a * b).sum() / denom)


@register
class PairsTrading(Strategy):
    """Mean-reversion on z-scored OLS-hedged spreads across a small pair list."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="pairs",
        name="Pairs Trading",
        description="OLS hedge ratio + z-score mean-reversion across hand-picked pairs.",
        universe=PAIRS_UNIVERSE,
        rebalance_frequency="weekly",
        enabled_live=True,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookback_days": 60,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_active_pairs": 5,
        "min_history_days": 90,
    }

    # Spec §2.3: entry z (1.5/2.0/2.5), exit z (0/0.5), discovery window.
    param_grid: ClassVar[dict[str, list[Any]]] = {
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.0, 0.5],
        "lookback_days": [45, 60, 90],
    }

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")
        # Held positions persist across rebalance days so we can apply exit-z logic.
        self._state: dict[
            tuple[str, str], int
        ] = {}  # value: -1 short-spread, +1 long-spread, 0 flat

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        return cls(bars=bars, params=params)

    def _spread_z(self, a: str, b: str, loc: int) -> tuple[float, float]:
        """Return (z, hedge) for the spread on the day at ``loc``."""
        if a not in self._close.columns or b not in self._close.columns:
            return float("nan"), float("nan")
        lookback = int(self.params["lookback_days"])
        win_a = self._close[a].iloc[max(loc - lookback, 0) : loc + 1].dropna()
        win_b = self._close[b].iloc[max(loc - lookback, 0) : loc + 1].dropna()
        common = win_a.index.intersection(win_b.index)
        if len(common) < 10:
            return float("nan"), float("nan")
        log_a = np.log(win_a.loc[common])
        log_b = np.log(win_b.loc[common])
        hedge = _hedge_ratio(log_a, log_b)
        if not np.isfinite(hedge):
            return float("nan"), float("nan")
        spread = log_a - hedge * log_b
        mu = float(spread.mean())
        sd = float(spread.std(ddof=1))
        if sd <= 0.0 or not np.isfinite(sd):
            return float("nan"), float("nan")
        latest_spread = float(spread.iloc[-1])
        z = (latest_spread - mu) / sd
        return z, hedge

    def generate_signals(self, asof: date) -> pd.Series:
        """Emit one signal per pair, encoded as ``"A/B": z``."""
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]):
            return pd.Series(dtype=float)
        signals: dict[str, float] = {}
        for a, b in SEED_PAIRS:
            z, _ = self._spread_z(a, b, loc)
            if np.isfinite(z):
                signals[f"{a}/{b}"] = z
        return pd.Series(signals)

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]) or equity <= 0:
            return {}

        entry_z = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        max_active = int(self.params["max_active_pairs"])

        decisions: list[tuple[tuple[str, str], int, float]] = []  # (pair, side, hedge)
        for a, b in SEED_PAIRS:
            z, hedge = self._spread_z(a, b, loc)
            if not np.isfinite(z) or not np.isfinite(hedge):
                self._state[(a, b)] = 0
                continue
            current = self._state.get((a, b), 0)
            new_side = current
            if current == 0:
                if z > entry_z:
                    new_side = -1  # short spread: sell A, buy hedge*B
                elif z < -entry_z:
                    new_side = +1
            elif abs(z) < exit_z:
                new_side = 0
            self._state[(a, b)] = new_side
            if new_side != 0:
                decisions.append(((a, b), new_side, hedge))

        if not decisions:
            return {}
        decisions = decisions[:max_active]

        per_pair_dollars = equity / len(decisions)
        weights: dict[str, float] = {}
        for (a, b), side, hedge in decisions:
            # Equal-dollar split per leg, then signed by side.
            half = (per_pair_dollars / 2.0) / equity
            weights[a] = weights.get(a, 0.0) + side * half
            weights[b] = weights.get(b, 0.0) + (-side) * half * float(np.sign(hedge) or 1.0)

        prices = self._close.iloc[loc].dropna()
        return size_to_shares(pd.Series(weights), prices, equity)
