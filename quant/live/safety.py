"""Pre-trade safety checks for the live rebalance path.

Three independent guards, each returning a ``CheckResult`` so the rebalance
orchestrator can log and skip selectively rather than crash:

  * ``check_market_open(asof)`` — is today a real NYSE trading day?
  * ``check_reconciliation(...)`` — does the Alpaca aggregate position book
    agree (within tolerance) with the sum of our per-strategy snapshots?
  * ``check_risk_limits(...)`` — has any strategy breached its max-drawdown
    circuit breaker? If so, that strategy is paused for today.

All three are pure-ish (no Alpaca calls inside the check fns; the caller
threads the inputs in) so the same harness exercises them in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from quant.execution.alpaca import PositionRow
from quant.live.bookkeeping import last_strategy_positions, read_equity
from quant.strategies import REGISTRY
from quant.util.trading_calendar import is_trading_day


@dataclass(frozen=True)
class CheckResult:
    """Result of a single safety check."""

    ok: bool
    name: str
    detail: str = ""


# --- market hours ---------------------------------------------------------


def check_market_open(asof: date) -> CheckResult:
    """Block the rebalance if today is not a trading day."""
    if not is_trading_day(asof):
        return CheckResult(
            ok=False,
            name="market_open",
            detail=f"{asof.isoformat()} is not a NYSE trading day; skipping rebalance.",
        )
    return CheckResult(ok=True, name="market_open", detail=f"{asof.isoformat()} is open")


# --- reconciliation -------------------------------------------------------


def _snapshot_aggregate(data_dir: Path, slugs: list[str]) -> dict[str, int]:
    """Sum per-strategy snapshots into expected aggregate positions."""
    out: dict[str, int] = {}
    for slug in slugs:
        snap = last_strategy_positions(data_dir, slug)
        for sym, qty in snap.items():
            out[sym] = out.get(sym, 0) + int(qty)
    return {k: v for k, v in out.items() if v != 0}


def check_reconciliation(
    *,
    data_dir: Path,
    alpaca_positions: list[PositionRow],
    enabled_slugs: list[str],
    winddown_slugs: list[str] | None = None,
    tolerance_shares: int = 1,
) -> CheckResult:
    """Compare ``sum(per-strategy snapshots)`` to Alpaca's aggregate book.

    Returns a non-OK result with a per-symbol diff string when the absolute
    difference exceeds ``tolerance_shares`` on any symbol. The first run after
    a fresh deploy has no snapshots → returns OK by design (nothing to compare).

    ``winddown_slugs`` lists orphan strategies currently being wound down; their
    last snapshot is counted as "expected" during convergence so that positions
    still held in Alpaca do not trigger a false reconciliation failure.
    """
    expected = _snapshot_aggregate(data_dir, list(enabled_slugs) + list(winddown_slugs or []))
    if not expected:
        return CheckResult(
            ok=True,
            name="reconciliation",
            detail="no prior snapshots; first run is implicitly reconciled",
        )

    actual: dict[str, int] = {p.symbol: int(p.qty) for p in alpaca_positions}

    diffs: list[tuple[str, int, int]] = []
    for sym in sorted(set(expected) | set(actual)):
        e = expected.get(sym, 0)
        a = actual.get(sym, 0)
        if abs(e - a) > tolerance_shares:
            diffs.append((sym, e, a))

    if diffs:
        detail = "; ".join(f"{s}: expected={e} actual={a}" for s, e, a in diffs[:10])
        return CheckResult(
            ok=False,
            name="reconciliation",
            detail=f"{len(diffs)} symbol diff(s) > {tolerance_shares}: {detail}",
        )
    return CheckResult(
        ok=True, name="reconciliation", detail=f"all {len(expected)} symbols within tolerance"
    )


# --- risk limits ----------------------------------------------------------


@dataclass(frozen=True)
class StrategyRiskBudget:
    """Per-strategy risk limits applied before each rebalance."""

    max_drawdown: float = 0.25  # halt the strategy if recent equity dd worse than this
    drawdown_lookback_days: int = 90
    max_position_pct: float = 0.20  # max single position as % of total equity


@dataclass(frozen=True)
class RiskCheckResult:
    """Aggregated risk-check result with per-strategy halt set."""

    ok: bool
    halted_strategies: frozenset[str]
    detail: str


def _recent_drawdown_pct(data_dir: Path, lookback_days: int) -> float:
    """Compute the worst peak-to-trough drawdown in the trailing ``lookback_days``."""
    equity_df = read_equity(data_dir)
    if equity_df.empty:
        return 0.0
    window = equity_df.tail(lookback_days)
    if window.empty:
        return 0.0
    equity = window["equity"].astype(float)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())  # non-positive


def check_risk_limits(
    *,
    data_dir: Path,
    enabled_slugs: list[str],
    budget: StrategyRiskBudget | None = None,
) -> RiskCheckResult:
    """Halt strategies whose drawdown is worse than the budget allows.

    Drawdown is computed against the *account-level* equity history (not per
    strategy). This is intentionally conservative: if the joint book has bled
    25%, halt every live-enabled strategy until we manually intervene. A
    future iteration can split equity attribution per strategy and apply the
    limit individually.
    """
    budget = budget or StrategyRiskBudget()
    dd = _recent_drawdown_pct(data_dir, budget.drawdown_lookback_days)
    halted: set[str] = set()
    if dd <= -abs(budget.max_drawdown):
        halted.update(enabled_slugs)
        return RiskCheckResult(
            ok=False,
            halted_strategies=frozenset(halted),
            detail=(
                f"account drawdown {dd:.2%} ≤ -{budget.max_drawdown:.2%}; "
                f"halting {len(enabled_slugs)} strategies"
            ),
        )
    return RiskCheckResult(
        ok=True,
        halted_strategies=frozenset(),
        detail=f"account drawdown {dd:.2%} within budget",
    )


# --- enabled-strategy helper ---------------------------------------------


def enabled_strategy_slugs() -> list[str]:
    """Return the sorted slugs of every strategy with ``enabled_live = True``."""
    return sorted(slug for slug, cls in REGISTRY.items() if cls.spec.enabled_live)


# --- bar staleness --------------------------------------------------------


def check_bar_freshness(
    data_dir: Path,
    *,
    symbols: list[str],
    asof: date,
    max_age_days: int = 4,
) -> CheckResult:
    """Verify that we have recent bars cached for at least one symbol.

    The threshold is intentionally generous (4 trading days) so a long-weekend
    + holiday combo doesn't false-alarm. The check pulls the most recent date
    across all symbols rather than per-symbol because the parquet cache only
    refreshes the union universe periodically.
    """
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        return CheckResult(
            ok=False,
            name="bar_freshness",
            detail=f"{raw_dir} does not exist — run `quant data refresh` first",
        )

    latest = None
    for symbol in symbols:
        path = raw_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        d = pd.Timestamp(df.index.max()).date()
        if latest is None or d > latest:
            latest = d

    if latest is None:
        return CheckResult(
            ok=False,
            name="bar_freshness",
            detail="no bar parquet files cached yet — run `quant data refresh`",
        )

    age = (asof - latest).days
    if age > max_age_days:
        return CheckResult(
            ok=False,
            name="bar_freshness",
            detail=f"latest bar is {latest.isoformat()} ({age} days old; threshold {max_age_days})",
        )
    return CheckResult(
        ok=True, name="bar_freshness", detail=f"latest bar {latest.isoformat()} ({age} days old)"
    )
