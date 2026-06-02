"""Portfolio-level risk analytics: VaR, CVaR, volatility, and factor exposure.

This is the institutional risk view the live path has been missing: rather than
only gross-exposure + single-name caps (see ``pretrade.py``), it characterizes
the *distribution* of portfolio P&L from a holdings vector and a returns panel.

It is intentionally a standalone ANALYSIS layer — it is NOT wired into the live
order-submission path (so it can never block a live rebalance). It powers
``quant risk portfolio`` and feeds the read-only Claude analyst brief. Promoting
any of these to hard pre-trade gates is a deliberate, reviewed follow-up.

The core ``compute_portfolio_risk`` is a pure function (weights + returns) so it
is trivially unit-testable with no data/network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.util.logging import logger

_TRADING_DAYS = 252

# Asset-class buckets for the live ETF universe, so a gate/brief can see
# concentration by asset class (not just per-symbol). Unknown symbols (e.g. the
# multi-factor single-stock universe) bucket to "other".
_SECTOR_MAP: dict[str, str] = {
    "SPY": "equity", "EFA": "equity", "EEM": "equity", "QQQ": "equity", "IWM": "equity",
    "VNQ": "real_estate",
    "TLT": "bond", "IEF": "bond", "SHY": "bond", "LQD": "bond", "HYG": "bond", "AGG": "bond",
    "GLD": "gold", "IAU": "gold",
    "DBC": "commodity", "USO": "commodity", "SLV": "commodity",
}


def _sector_exposure(weights: dict[str, float]) -> dict[str, float]:
    """Group |weight| into asset-class buckets via ``_SECTOR_MAP`` (unknown -> 'other')."""
    out: dict[str, float] = {}
    for sym, w in weights.items():
        bucket = _SECTOR_MAP.get(sym.upper(), "other")
        out[bucket] = out.get(bucket, 0.0) + abs(float(w))
    return {k: v for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


@dataclass(frozen=True)
class PortfolioRisk:
    """Risk characterization of a holdings vector. Fractions are of equity."""

    n_positions: int
    gross_exposure: float  # sum |weight|
    net_exposure: float  # sum weight
    ann_vol: float | None  # annualized portfolio volatility
    var_95: float | None  # 1-day historical VaR at 95% (positive = loss)
    cvar_95: float | None  # 1-day historical CVaR/expected-shortfall at 95%
    beta_to_benchmark: float | None  # OLS beta of portfolio vs benchmark
    top_name_weight: float | None  # largest single-name |weight|
    lookback_days: int
    # Asset-class concentration (bucket -> sum |weight|), and a fail-state flag so a
    # future fail-closed gate can tell "within limits" from "could not compute".
    sector_exposure: dict[str, float] = field(default_factory=dict)
    computable: bool = True  # False => the distributional metrics could not be computed
    degraded_metrics: tuple[str, ...] = ()  # names of metrics that came back None

    def render(self) -> str:
        """Compact one-block summary for Slack/CLI/the analyst brief."""

        def _pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.2%}"

        def _num(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.2f}"

        line = (
            f"positions {self.n_positions} | gross {self.gross_exposure:.0%} "
            f"net {self.net_exposure:+.0%} | ann vol {_pct(self.ann_vol)} | "
            f"1d VaR95 {_pct(self.var_95)} CVaR95 {_pct(self.cvar_95)} | "
            f"beta {_num(self.beta_to_benchmark)} | top {_pct(self.top_name_weight)}"
        )
        if self.sector_exposure:
            buckets = ", ".join(f"{k} {v:.0%}" for k, v in self.sector_exposure.items())
            line += f" | by class: {buckets}"
        return line


def compute_portfolio_risk(
    weights: dict[str, float],
    returns: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    confidence: float = 0.95,
) -> PortfolioRisk:
    """Characterize the risk of a holdings vector from a daily-returns panel.

    ``weights`` maps symbol -> signed portfolio weight (fraction of equity).
    ``returns`` is a daily simple-returns panel (rows = dates, cols = symbols).
    ``benchmark`` (optional) is a daily-returns series for a beta computation.
    Missing symbols / degenerate inputs degrade the affected metric to ``None``
    rather than raising.
    """
    nonzero = {k: float(v) for k, v in weights.items() if abs(float(v)) > 0}
    gross = float(sum(abs(v) for v in nonzero.values()))
    net = float(sum(nonzero.values()))
    top = max((abs(v) for v in nonzero.values()), default=None)
    sector = _sector_exposure(nonzero)

    base = PortfolioRisk(
        n_positions=len(nonzero),
        gross_exposure=gross,
        net_exposure=net,
        ann_vol=None,
        var_95=None,
        cvar_95=None,
        beta_to_benchmark=None,
        top_name_weight=top,
        lookback_days=0,
        sector_exposure=sector,
        computable=False,  # no usable returns panel => distributional metrics absent
        degraded_metrics=("ann_vol", "var_95", "cvar_95", "beta_to_benchmark"),
    )

    if returns is None or returns.empty or not nonzero:
        return base

    cols = [s for s in nonzero if s in returns.columns]
    if not cols:
        return base
    w = pd.Series({s: nonzero[s] for s in cols}, dtype=float)
    panel = returns[cols].dropna(how="all")
    if panel.empty:
        return base
    port = panel.fillna(0.0).mul(w, axis=1).sum(axis=1)
    port = port.replace([np.inf, -np.inf], np.nan).dropna()
    if len(port) < 2:
        return base

    ann_vol = float(port.std(ddof=1) * math.sqrt(_TRADING_DAYS))
    q = float(np.quantile(port.to_numpy(), 1.0 - confidence))
    var = float(-q)  # positive => a loss
    tail = port[port <= q]
    cvar = float(-tail.mean()) if len(tail) > 0 else var

    beta: float | None = None
    if benchmark is not None and not benchmark.empty:
        joined = pd.concat([port.rename("p"), benchmark.rename("b")], axis=1).dropna()
        if len(joined) >= 2:
            var_b = float(joined["b"].var(ddof=1))
            if var_b > 0:
                cov_pb = float(joined["p"].cov(joined["b"]))
                beta = cov_pb / var_b

    return PortfolioRisk(
        n_positions=len(nonzero),
        gross_exposure=gross,
        net_exposure=net,
        ann_vol=ann_vol,
        var_95=var,
        cvar_95=cvar,
        beta_to_benchmark=beta,
        top_name_weight=top,
        lookback_days=len(port),
        sector_exposure=sector,
        computable=True,
        degraded_metrics=() if beta is not None else ("beta_to_benchmark",),
    )


def weights_from_positions(
    positions: dict[str, int],
    prices: dict[str, float],
    equity: float,
) -> dict[str, float]:
    """Convert share quantities + reference prices into signed equity weights."""
    if equity <= 0:
        return {}
    out: dict[str, float] = {}
    for sym, qty in positions.items():
        px = float(prices.get(sym, 0.0))
        if px > 0 and qty != 0:
            out[sym] = (qty * px) / equity
    return out


def live_portfolio_risk(
    positions: dict[str, int],
    equity: float,
    *,
    asof: date,
    lookback_days: int = 180,
    benchmark_symbol: str = "SPY",
) -> PortfolioRisk | None:
    """Best-effort: fetch recent bars for the held names + benchmark and compute
    portfolio risk. Returns ``None`` on no positions or any data failure — this is
    an analysis convenience and must never raise into a caller's hot path.
    """
    if not positions or equity <= 0:
        return None
    try:
        from quant.data.bars import BarRequest, get_bars
        from quant.strategies._common import field_frame

        symbols = sorted(set(positions) | {benchmark_symbol})
        start = asof - timedelta(days=lookback_days * 2)  # calendar pad → ~lookback trading days
        bars = get_bars(BarRequest(symbols=symbols, start=start, end=asof))
        if bars.empty:
            return None
        close = field_frame(bars, "close")
        returns = close.pct_change(fill_method=None).dropna(how="all").tail(lookback_days)
        prices: dict[str, float] = {}
        for sym in close.columns:
            col = close[sym].dropna()
            if sym in positions and not col.empty:
                prices[sym] = float(col.iloc[-1])
        weights = weights_from_positions(positions, prices, equity)
        benchmark = returns[benchmark_symbol] if benchmark_symbol in returns.columns else None
        return compute_portfolio_risk(weights, returns, benchmark=benchmark)
    except Exception as exc:  # analysis convenience — never raise
        logger.info("live_portfolio_risk skipped ({!r})", exc)
        return None
