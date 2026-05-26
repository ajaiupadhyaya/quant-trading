"""Daily rebalance orchestrator: signals -> orders -> bookkeeping.

Algorithm per run:

  1. Pull Alpaca account snapshot (equity, cash, buying power).
  2. Snapshot equity into ``data/live/equity.parquet``.
  3. For each strategy with ``spec.enabled_live`` True:
       a. Fetch bars over the trailing 2 years (enough for any 252-day warmup).
       b. Allocate this strategy's slice of equity (equal split across enabled).
       c. Build the strategy and ask for target shares.
       d. Reconcile against our last per-strategy snapshot (NOT Alpaca's
          aggregate positions — those are the union across all strategies).
       e. Submit each delta order via Alpaca with the per-strategy client_order_id.
       f. Append every submitted order to ``data/live/trades.parquet``.
       g. Snapshot the new per-strategy positions to ``strategy_positions.parquet``.

Both real and dry-run modes go through the same code path; dry-run flips the
``submit_order`` flag and prints to logs instead of hitting the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data.bars import BarRequest, get_bars
from quant.execution.alpaca import AlpacaClient
from quant.execution.orders import OrderTemplate
from quant.execution.reconciler import reconcile
from quant.live.bookkeeping import (
    append_equity_row,
    append_trades,
    last_strategy_positions,
    write_strategy_positions,
)
from quant.strategies import REGISTRY
from quant.strategies.base import Strategy
from quant.util.config import Settings
from quant.util.logging import logger


@dataclass
class StrategyRebalanceOutcome:
    slug: str
    target: dict[str, int]
    previous: dict[str, int]
    orders: list[OrderTemplate]
    error: str | None = None


@dataclass
class RebalanceReport:
    asof: date
    equity: float
    enabled_strategies: list[str]
    outcomes: list[StrategyRebalanceOutcome] = field(default_factory=list)
    dry_run: bool = False
    safety_checks: list[Any] = field(default_factory=list)
    halted_strategies: frozenset[str] = frozenset()
    skipped_reason: str | None = None

    @property
    def total_orders(self) -> int:
        return sum(len(o.orders) for o in self.outcomes)


def _enabled_strategies() -> list[str]:
    return sorted(slug for slug, cls in REGISTRY.items() if cls.spec.enabled_live)


def _governance_selected_strategies(
    data_dir: Path,
    *,
    include_quarantined: bool,
) -> tuple[list[str], str | None]:
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import load_strategy_states, strategy_states_path

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError as exc:
        return [], f"governance unavailable: {exc}. Run `quant governance refresh`."

    selected: list[str] = []
    for slug, state in sorted(states.items()):
        if slug not in REGISTRY:
            continue
        if state.state is GovernanceState.LIVE:
            selected.append(slug)
        elif include_quarantined and state.state is GovernanceState.QUARANTINED:
            selected.append(slug)
    return selected, None


def _bars_for(strategy_cls: type[Strategy], asof: date, history_days: int) -> pd.DataFrame:
    start = asof - timedelta(days=history_days)
    req = BarRequest(symbols=list(strategy_cls.spec.universe), start=start, end=asof)
    return get_bars(req)


def _latest_chosen_params(data_dir: Path, slug: str) -> dict[str, Any]:
    """Read ``data/backtests/<slug>/chosen_params.json`` and return its ``latest`` field.

    Returns an empty dict if the file is missing or doesn't contain ``latest`` —
    in that case the strategy will simply run with its own ``default_params``,
    so live trading never silently breaks just because the nightly backtest
    artifact is stale or missing.
    """
    path = data_dir / "backtests" / slug / "chosen_params.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # corrupt JSON shouldn't crash the rebalance
        logger.warning("Could not read {} ({}); falling back to defaults", path, exc)
        return {}
    latest = payload.get("latest", {})
    if not isinstance(latest, dict):
        return {}
    return latest


def run_rebalance(
    *,
    asof: date | None = None,
    dry_run: bool = False,
    history_days: int = 1100,
    client: AlpacaClient | None = None,
    settings: Settings | None = None,
    strategies: list[str] | None = None,
    skip_safety_checks: bool = False,
    risk_budget: object | None = None,
    include_quarantined: bool = False,
) -> RebalanceReport:
    """Execute one rebalance pass. Returns a structured report for the CLI to render."""
    from quant.live.safety import (
        StrategyRiskBudget,
        check_market_open,
        check_reconciliation,
        check_risk_limits,
    )

    settings = settings or Settings()  # type: ignore[call-arg]
    client = client or AlpacaClient(settings=settings)
    asof = asof or date.today()

    safety_results: list[Any] = []

    if include_quarantined and not dry_run:
        return RebalanceReport(
            asof=asof,
            equity=0.0,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            skipped_reason="--include-quarantined is allowed only for dry-run observation.",
        )

    # Guard 1: is today a trading day at all? Skip everything if not.
    if not skip_safety_checks:
        market_check = check_market_open(asof)
        safety_results.append(market_check)
        if not market_check.ok:
            logger.warning("safety: {} — {}", market_check.name, market_check.detail)
            return RebalanceReport(
                asof=asof,
                equity=0.0,
                enabled_strategies=[],
                outcomes=[],
                dry_run=dry_run,
                safety_checks=safety_results,
                skipped_reason=market_check.detail,
            )

    account = client.account()
    append_equity_row(
        settings.data_dir,
        asof=asof,
        equity=account.equity,
        last_equity=account.last_equity,
        cash=account.cash,
        buying_power=account.buying_power,
        portfolio_value=account.portfolio_value,
    )

    if strategies is not None:
        enabled = strategies
    else:
        enabled, governance_error = _governance_selected_strategies(
            settings.data_dir,
            include_quarantined=include_quarantined,
        )
        if governance_error is not None:
            logger.error("{}", governance_error)
            return RebalanceReport(
                asof=asof,
                equity=account.equity,
                enabled_strategies=[],
                outcomes=[],
                dry_run=dry_run,
                safety_checks=safety_results,
                skipped_reason=governance_error,
            )
    if not enabled:
        logger.warning("No live-enabled strategies; rebalance is a no-op.")
        return RebalanceReport(
            asof=asof,
            equity=account.equity,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            safety_checks=safety_results,
        )

    # Guard 2: reconciliation against Alpaca's aggregate position book.
    halted: frozenset[str] = frozenset()
    if not skip_safety_checks:
        recon = check_reconciliation(
            data_dir=settings.data_dir,
            alpaca_positions=client.positions(),
            enabled_slugs=enabled,
        )
        safety_results.append(recon)
        if not recon.ok:
            logger.error("safety: reconciliation MISMATCH — refusing to trade. {}", recon.detail)
            return RebalanceReport(
                asof=asof,
                equity=account.equity,
                enabled_strategies=enabled,
                outcomes=[],
                dry_run=dry_run,
                safety_checks=safety_results,
                skipped_reason=recon.detail,
            )

        # Guard 3: risk-limit circuit breaker on the account-level equity history.
        budget_obj = (
            risk_budget if isinstance(risk_budget, StrategyRiskBudget) else StrategyRiskBudget()
        )
        risk = check_risk_limits(
            data_dir=settings.data_dir, enabled_slugs=enabled, budget=budget_obj
        )
        safety_results.append(risk)
        if not risk.ok:
            logger.error(
                "safety: risk circuit breaker tripped — {} strategies halted. {}",
                len(risk.halted_strategies),
                risk.detail,
            )
            halted = risk.halted_strategies

    per_strategy_equity = account.equity / len(enabled)
    report = RebalanceReport(
        asof=asof,
        equity=account.equity,
        enabled_strategies=enabled,
        dry_run=dry_run,
        safety_checks=safety_results,
        halted_strategies=halted,
    )

    all_trade_rows: list[dict[str, object]] = []

    for slug in enabled:
        if slug in halted:
            report.outcomes.append(
                StrategyRebalanceOutcome(
                    slug=slug,
                    target={},
                    previous=last_strategy_positions(settings.data_dir, slug),
                    orders=[],
                    error="halted by risk circuit breaker",
                )
            )
            continue

        if slug not in REGISTRY:
            report.outcomes.append(
                StrategyRebalanceOutcome(
                    slug=slug,
                    target={},
                    previous={},
                    orders=[],
                    error=f"strategy {slug!r} not registered",
                )
            )
            continue

        strategy_cls = REGISTRY[slug]
        try:
            bars = _bars_for(strategy_cls, asof, history_days)
        except Exception as exc:
            report.outcomes.append(
                StrategyRebalanceOutcome(
                    slug=slug,
                    target={},
                    previous=last_strategy_positions(settings.data_dir, slug),
                    orders=[],
                    error=f"bar fetch failed: {exc!r}",
                )
            )
            continue

        if bars.empty:
            report.outcomes.append(
                StrategyRebalanceOutcome(
                    slug=slug,
                    target={},
                    previous=last_strategy_positions(settings.data_dir, slug),
                    orders=[],
                    error="no bars returned",
                )
            )
            continue

        chosen = _latest_chosen_params(settings.data_dir, slug)
        if chosen:
            logger.info("Using chosen_params.json[latest] for {}: {}", slug, chosen)
        strategy = strategy_cls.build(bars=bars, params=chosen or None)
        try:
            target = strategy.target_positions(asof, per_strategy_equity)
        except Exception as exc:
            logger.exception("strategy {} target_positions raised", slug)
            target = {}
            err: str | None = f"target_positions raised: {exc!r}"
        else:
            err = None

        previous = last_strategy_positions(settings.data_dir, slug)
        orders = reconcile(target=target, current=previous, strategy_slug=slug)

        for order in orders:
            try:
                coid = client.submit_order(order, dry_run=dry_run)
            except Exception as exc:
                logger.exception("submit_order failed for {} {}", slug, order.symbol)
                err = f"submit_order failed: {exc!r}"
                continue
            all_trade_rows.append(
                {
                    "date": pd.Timestamp(asof),
                    "strategy": slug,
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "qty": int(order.qty),
                    "client_order_id": coid,
                    "dry_run": bool(dry_run),
                }
            )

        report.outcomes.append(
            StrategyRebalanceOutcome(
                slug=slug,
                target=target,
                previous=previous,
                orders=orders,
                error=err,
            )
        )

        # Update our per-strategy bookkeeping with the new target, even in dry-run.
        # Dry-run snapshots are useful in tests but in production daily-rebalance.yml
        # we only commit when dry_run=False.
        if target and not dry_run:
            write_strategy_positions(settings.data_dir, asof, slug, target)

    if all_trade_rows:
        append_trades(settings.data_dir, all_trade_rows)

    return report
