"""Scenario / stress-shock evaluation for the live book (Raise-the-Ceiling Phase 2).

How much would today's holdings lose under a 2008-style crash, a +100bp rate
shock, etc.? Both scenario kinds reduce to the same kernel — apply a per-asset
return shock to current signed weights -> portfolio P&L%:

    pnl_pct = sum_i  weight_i * shock_i

This is a standalone, READ-ONLY analysis layer (mirrors ``portfolio.py``): it is
WARN-only, never wired to block an order, and fail-open at every live entry point.
``compute_stress`` is pure (weights + returns/shocks) so it is trivially testable.

Shock-key convention: UPPERCASE keys are symbols (``"TLT"``), lowercase keys are
asset-class buckets (``"bond"``, matching ``portfolio._SECTOR_MAP`` values). A
symbol-level shock overrides its class-level shock for that symbol.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.risk.portfolio import _SECTOR_MAP, weights_from_positions
from quant.util.logging import logger


@dataclass(frozen=True)
class HistoricalScenario:
    """Replay each held asset's OWN cumulative simple return over [start, end]."""

    name: str
    start: date
    end: date
    description: str = ""
    kind: str = "historical"


@dataclass(frozen=True)
class HypotheticalScenario:
    """Apply a shock vector (asset-class bucket OR symbol -> return shock)."""

    name: str
    shocks: dict[str, float]
    description: str = ""
    kind: str = "hypothetical"


Scenario = HistoricalScenario | HypotheticalScenario


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    kind: str
    pnl_pct: float | None  # signed; negative = loss; None if nothing computable
    by_class: dict[str, float] = field(default_factory=dict)  # per-class P&L contribution
    missing_symbols: tuple[str, ...] = ()  # held names with no shock/no data
    computable: bool = True


@dataclass(frozen=True)
class StressReport:
    results: tuple[ScenarioResult, ...]
    worst_loss: float | None  # -(min pnl over computable); positive = loss
    worst_scenario: str | None
    computable: bool  # at least one scenario computable
    degraded: tuple[str, ...] = ()  # scenario names that were not computable

    def render(self) -> str:
        """Compact one-block summary for CLI/brief/Slack."""
        if not self.computable or self.worst_loss is None:
            return "stress: n/a (no computable scenarios)"
        head = f"worst {self.worst_scenario} {self.worst_loss:+.1%} loss"
        parts = []
        for r in self.results:
            if r.pnl_pct is None:
                parts.append(f"{r.name} n/a")
            else:
                parts.append(f"{r.name} {r.pnl_pct:+.1%}")
        return head + " | " + ", ".join(parts)


def _historical_shock(sym: str, returns: pd.DataFrame, start: date, end: date) -> float | None:
    """Cumulative simple return of ``sym`` over [start, end], or None if no data."""
    if returns is None or returns.empty or sym not in returns.columns:
        return None
    col = returns[sym]
    mask = (col.index >= pd.Timestamp(start)) & (col.index <= pd.Timestamp(end))
    window = col[mask].dropna()
    if window.empty:
        return None
    return float(np.prod(1.0 + window.to_numpy()) - 1.0)


def _hypothetical_shock(sym: str, shocks: dict[str, float]) -> float | None:
    """Symbol-level shock overrides class-level; None if neither present."""
    if sym.upper() in shocks:
        return float(shocks[sym.upper()])
    bucket = _SECTOR_MAP.get(sym.upper(), "other")
    if bucket in shocks:
        return float(shocks[bucket])
    return None


def _shock_for(scen: Scenario, sym: str, returns: pd.DataFrame) -> float | None:
    """Resolve one scenario's shock for one symbol (None => no shock/no data)."""
    if isinstance(scen, HistoricalScenario):
        return _historical_shock(sym, returns, scen.start, scen.end)
    return _hypothetical_shock(sym, scen.shocks)


def _evaluate(scen: Scenario, weights: dict[str, float], returns: pd.DataFrame) -> ScenarioResult:
    """Apply ``scen``'s shocks across ``weights``. An asset whose shock is None
    contributes 0 and is recorded in ``missing_symbols``."""
    pnl = 0.0
    by_class: dict[str, float] = {}
    missing: list[str] = []
    any_shocked = False
    for sym, w in weights.items():
        s = _shock_for(scen, sym, returns)
        if s is None:
            missing.append(sym)
            s = 0.0
        else:
            any_shocked = True
        contrib = float(w) * s
        pnl += contrib
        bucket = _SECTOR_MAP.get(sym.upper(), "other")
        by_class[bucket] = by_class.get(bucket, 0.0) + contrib
    return ScenarioResult(
        name=scen.name,
        kind=scen.kind,
        pnl_pct=pnl if any_shocked else None,
        by_class={k: v for k, v in sorted(by_class.items(), key=lambda kv: kv[1])},
        missing_symbols=tuple(sorted(missing)),
        computable=any_shocked,
    )


def compute_stress(
    weights: dict[str, float],
    returns: pd.DataFrame,
    scenarios: Iterable[Scenario],
) -> StressReport:
    """Pure: evaluate ``scenarios`` against ``weights`` (+ ``returns`` for historical)."""
    nonzero = {k: float(v) for k, v in weights.items() if abs(float(v)) > 0}
    results: list[ScenarioResult] = []
    degraded: list[str] = []
    for scen in scenarios:
        res = _evaluate(scen, nonzero, returns)
        results.append(res)
        if not res.computable:
            degraded.append(scen.name)

    computable_pnls = [(r.name, r.pnl_pct) for r in results if r.pnl_pct is not None]
    if not computable_pnls:
        return StressReport(
            results=tuple(results),
            worst_loss=None,
            worst_scenario=None,
            computable=False,
            degraded=tuple(degraded),
        )
    worst_name, worst_pnl = min(computable_pnls, key=lambda kv: kv[1])
    return StressReport(
        results=tuple(results),
        worst_loss=-worst_pnl,
        worst_scenario=worst_name,
        computable=True,
        degraded=tuple(degraded),
    )


def default_scenarios() -> tuple[Scenario, ...]:
    """Curated stress library (see spec). Windows are peak->trough of the episode."""
    return (
        HistoricalScenario(
            "2008-GFC", date(2008, 9, 1), date(2009, 3, 9), "Lehman to the market bottom"
        ),
        HistoricalScenario("2020-COVID", date(2020, 2, 19), date(2020, 3, 23), "COVID crash"),
        HistoricalScenario(
            "2022-rate-selloff",
            date(2022, 1, 1),
            date(2022, 10, 14),
            "2022 rate-driven 60/40 drawdown",
        ),
        HistoricalScenario(
            "2013-taper-tantrum", date(2013, 5, 22), date(2013, 6, 24), "Taper tantrum"
        ),
        HypotheticalScenario(
            "equity-crash-20",
            {"equity": -0.20, "real_estate": -0.25, "commodity": -0.10, "gold": 0.05, "bond": 0.05},
            "Broad equity crash, flight to quality",
        ),
        HypotheticalScenario(
            "rate-shock-+100bp",
            {
                "TLT": -0.15,
                "IEF": -0.07,
                "equity": -0.05,
                "real_estate": -0.10,
                "gold": -0.03,
                "bond": -0.07,
            },
            "+100bp parallel rate shock (duration-aware)",
        ),
        HypotheticalScenario(
            "stagflation",
            {"commodity": 0.15, "gold": 0.10, "bond": -0.10, "equity": -0.10},
            "Inflationary stagnation",
        ),
        HypotheticalScenario(
            "risk-off-flight",
            {"equity": -0.15, "gold": 0.08, "bond": 0.05, "commodity": -0.10, "real_estate": -0.12},
            "Risk-off flight to safety",
        ),
    )


def live_stress(
    positions: dict[str, int],
    equity: float,
    *,
    asof: date,
    lookback_days: int = 180,
    scenarios: Iterable[Scenario] | None = None,
) -> StressReport | None:
    """Best-effort: fetch enough history to cover the historical windows + current
    prices, then run ``compute_stress``. Returns None on flat book / data failure —
    analysis convenience, must never raise into a caller's hot path.
    """
    if not positions or equity <= 0:
        return None
    scens = tuple(scenarios) if scenarios is not None else default_scenarios()
    try:
        from quant.data.bars import BarRequest, get_bars
        from quant.strategies._common import field_frame

        symbols = sorted(set(positions))
        # Fetch from the earliest historical-scenario start (pad) through asof.
        hist_starts = [s.start for s in scens if isinstance(s, HistoricalScenario)]
        earliest = min(hist_starts) if hist_starts else asof - timedelta(days=lookback_days * 2)
        start = min(earliest, asof - timedelta(days=lookback_days * 2)) - timedelta(days=7)
        bars = get_bars(BarRequest(symbols=symbols, start=start, end=asof))
        if bars.empty:
            return None
        close = field_frame(bars, "close")
        returns = close.pct_change(fill_method=None)
        prices: dict[str, float] = {}
        for sym in close.columns:
            col = close[sym].dropna()
            if sym in positions and not col.empty:
                prices[sym] = float(col.iloc[-1])
        weights = weights_from_positions(positions, prices, equity)
        return compute_stress(weights, returns, scens)
    except Exception as exc:  # analysis convenience — never raise
        logger.info("live_stress skipped ({!r})", exc)
        return None
