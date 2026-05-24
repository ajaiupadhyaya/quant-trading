"""Hierarchical Risk Parity (HRP) all-weather portfolio.

Lopez de Prado 2016: cluster assets by return correlation, then allocate
capital via recursive bisection so each side of every split contributes equal
risk. The result is a diversified, long-only weighting that avoids the
condition-number pathologies of mean-variance optimization on correlated
assets. Gross exposure is scaled to ``vol_target_annual`` using realized
portfolio vol over the lookback window.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from quant.data.universe import etf_universe
from quant.strategies import register
from quant.strategies._common import asof_index, field_frame, size_to_shares
from quant.strategies.base import Strategy, StrategySpec


def _seriation(link: np.ndarray, num_items: int, cur: int) -> list[int]:
    """Reorder leaves so distance between consecutive leaves is minimized."""
    if cur < num_items:
        return [cur]
    left = int(link[cur - num_items, 0])
    right = int(link[cur - num_items, 1])
    return _seriation(link, num_items, left) + _seriation(link, num_items, right)


def _ivp_weights(cov_slice: pd.DataFrame) -> pd.Series:
    """Inverse-variance portfolio weights, given a covariance slice."""
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp = ivp / ivp.sum()
    return pd.Series(ivp, index=cov_slice.index)


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items]
    w = np.asarray(_ivp_weights(sub).values).reshape(-1, 1)
    return float((w.T @ sub.values @ w).item())


def hrp_weights(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """Compute HRP weights from cov + corr matrices over the same symbols."""
    if cov.empty or cov.shape[0] < 2:
        if cov.shape[0] == 1:
            return pd.Series([1.0], index=cov.index)
        return pd.Series(dtype=float)

    dist = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    n = cov.shape[0]
    sort_idx = _seriation(link, n, 2 * n - 2)
    sorted_symbols = [cov.index[i] for i in sort_idx]

    weights = pd.Series(1.0, index=sorted_symbols)
    clusters: list[list[str]] = [sorted_symbols]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left = cluster[:half]
            right = cluster[half:]
            v_left = _cluster_variance(cov, left)
            v_right = _cluster_variance(cov, right)
            total = v_left + v_right
            alpha = 1.0 - (v_left / total) if total > 0 else 0.5
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    out: pd.Series = weights.reindex(cov.index).fillna(0.0)
    return out


@register
class RiskParity(Strategy):
    """HRP-weighted multi-asset portfolio, monthly rebalance, vol-targeted."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="risk-parity",
        name="HRP All-Weather",
        description="Hierarchical Risk Parity on ETF universe with constant-vol targeting.",
        universe=etf_universe(),
        rebalance_frequency="monthly",
        enabled_live=True,
    )

    default_params: ClassVar[dict[str, Any]] = {
        "lookback_days": 252,
        "vol_target_annual": 0.10,
        "max_leverage": 1.0,
        "min_history_days": 252,
    }

    def __init__(self, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._close = field_frame(bars, "close")
        self._returns = self._close.pct_change(fill_method=None)

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        return cls(bars=bars, params=params)

    def _weights_at(self, loc: int) -> pd.Series:
        lookback = int(self.params["lookback_days"])
        window = self._returns.iloc[max(loc - lookback, 0) : loc + 1].dropna(how="all", axis=1)
        window = window.dropna(axis=1, thresh=int(0.8 * len(window)))
        if window.shape[1] < 2 or window.shape[0] < 20:
            return pd.Series(dtype=float)
        cov = window.cov()
        corr = window.corr().fillna(0.0)
        weights = hrp_weights(cov, corr)
        if weights.sum() <= 0:
            return pd.Series(dtype=float)
        weights = weights / weights.sum()

        # Vol-target scaling: realized portfolio vol vs target
        port_returns = (window * weights).sum(axis=1)
        realized_vol = float(port_returns.std(ddof=1)) * float(np.sqrt(252))
        target = float(self.params["vol_target_annual"])
        if realized_vol > 0:
            weights = weights * (target / realized_vol)
        gross = float(weights.abs().sum())
        max_lev = float(self.params["max_leverage"])
        if gross > max_lev > 0:
            weights = weights * (max_lev / gross)
        out: pd.Series = weights
        return out

    def generate_signals(self, asof: date) -> pd.Series:
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None or loc < int(self.params["min_history_days"]):
            return pd.Series(dtype=float)
        return self._weights_at(loc)

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        weights = self.generate_signals(asof)
        if weights.empty or equity <= 0:
            return {}
        history = pd.DatetimeIndex(self._close.index)
        loc = asof_index(history, asof)
        if loc is None:
            return {}
        prices = self._close.iloc[loc].dropna()
        return size_to_shares(weights, prices, equity)
