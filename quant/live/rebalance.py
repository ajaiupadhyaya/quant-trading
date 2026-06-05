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
from quant.execution.netting import net_orders
from quant.execution.orders import OrderSide, OrderTemplate
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
    reference_prices: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class WindDownOutcome:
    slug: str
    exited: dict[str, int] = field(default_factory=dict)
    remaining: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
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
    winddown_outcomes: list[WindDownOutcome] = field(default_factory=list)

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
        if state.state is GovernanceState.LIVE or (
            include_quarantined and state.state is GovernanceState.QUARANTINED
        ):
            selected.append(slug)
    return selected, None


def _bars_for(strategy_cls: type[Strategy], asof: date, history_days: int) -> pd.DataFrame:
    start = asof - timedelta(days=history_days)
    req = BarRequest(symbols=list(strategy_cls.spec.universe), start=start, end=asof)
    return get_bars(req)


def _latest_reference_prices(bars: pd.DataFrame, symbols: set[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol in sorted(symbols):
        try:
            closes = bars[(symbol, "close")].dropna()
        except (KeyError, TypeError):
            continue
        if not closes.empty:
            prices[symbol] = float(closes.iloc[-1])
    return prices


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


def already_traded_today(client: object, asof: date) -> bool:
    """True iff the broker already has orders dated ``asof`` (idempotency guard).

    Uses a duck-typed ``list_orders_for_date(asof) -> list`` so tests can inject a
    fake. Defaults to False (fail-open to allow the trade) only if the client does
    not expose the method — but logs a warning, because without it the deterministic
    client_order_id (Task 9) is the sole double-submit backstop.
    """
    lister = getattr(client, "list_orders_for_date", None)
    if lister is None:
        logger.warning("client has no list_orders_for_date; relying on deterministic COID only")
        return False
    return len(lister(asof)) > 0


def _write_portfolio_risk_gate_artifact(data_dir: Path, *, asof: date, gate: Any) -> None:
    """Write a per-run portfolio-risk-gate artifact (atomic JSON). Best-effort —
    called from inside Guard 5's try/except so a write failure cannot escape."""
    from quant.util.atomic import write_json_atomic

    r = gate.risk
    payload = {
        "asof": asof.isoformat(),
        "mode": str(gate.mode),
        "ok": gate.ok,
        "severity": gate.severity,
        "violations": [
            {"code": v.code, "detail": v.detail, "bucket": v.bucket} for v in gate.violations
        ],
        "risk": {
            "n_positions": r.n_positions,
            "gross_exposure": r.gross_exposure,
            "net_exposure": r.net_exposure,
            "ann_vol": r.ann_vol,
            "var_95": r.var_95,
            "cvar_95": r.cvar_95,
            "beta_to_benchmark": r.beta_to_benchmark,
            "top_name_weight": r.top_name_weight,
            "lookback_days": r.lookback_days,
            "sector_exposure": dict(r.sector_exposure),
            "computable": r.computable,
            "degraded_metrics": list(r.degraded_metrics),
        },
    }
    path = data_dir / "risk" / f"portfolio_risk_gate.{asof.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


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
    record_bookkeeping: bool = True,
    winddown_participation: float = 0.10,
) -> RebalanceReport:
    """Execute one rebalance pass. Returns a structured report for the CLI to render."""
    from quant.live.safety import (
        CheckResult,
        StrategyRiskBudget,
        check_market_open,
        check_reconciliation,
        check_risk_limits,
    )

    settings = settings or Settings()  # type: ignore[call-arg]
    client = client or AlpacaClient(settings=settings)
    asof = asof or date.today()

    safety_results: list[Any] = []

    from quant.governance.halt import load_halt

    halt = load_halt(settings.data_dir)
    if halt.active and not dry_run:
        reason = f"Emergency halt active: {halt.reason}"
        logger.error(reason)
        return RebalanceReport(
            asof=asof,
            equity=0.0,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            safety_checks=safety_results,
            skipped_reason=reason,
        )

    if not dry_run and already_traded_today(client, asof):
        reason = f"orders already exist for {asof}; refusing to re-submit (idempotency)"
        logger.warning(reason)
        return RebalanceReport(
            asof=asof,
            equity=0.0,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            safety_checks=safety_results,
            skipped_reason=reason,
        )

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
    # NOTE: this records the REAL account equity series and is intentionally
    # written on dry-run too (see test_dry_run_does_not_persist_strategy_positions)
    # so the drawdown/drift guardrails have continuous history during the
    # shakedown. It is observability, not a faked trade — trades/positions below
    # are gated on `not dry_run`. The guard's equity-health guardrail
    # (not this write) is what distinguishes a dead feed from a flat book.
    if record_bookkeeping:
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
        reason = "No governance-live strategies; rebalance is fail-closed with no orders."
        logger.warning(reason)
        # Known limitation: wind-down currently requires >=1 live strategy to run;
        # if no strategy is live the rebalance returns before wind-down.
        return RebalanceReport(
            asof=asof,
            equity=account.equity,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            safety_checks=safety_results,
            skipped_reason=reason,
        )

    from quant.live.winddown import detect_orphans, winddown_orders

    orphans = detect_orphans(settings.data_dir)

    # Guard 2: reconciliation against Alpaca's aggregate position book.
    halted: frozenset[str] = frozenset()
    alpaca_positions: list[Any] = []  # captured once for Guard 5 post-trade reconstruction
    if not skip_safety_checks:
        alpaca_positions = client.positions()
        recon = check_reconciliation(
            data_dir=settings.data_dir,
            alpaca_positions=alpaca_positions,
            enabled_slugs=enabled,
            winddown_slugs=orphans,
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

    from quant.governance.allocation import allocate_capital
    from quant.governance.store import (
        load_strategy_states,
        load_validation_manifest,
        strategy_states_path,
        validation_manifest_path,
    )

    try:
        allocation = allocate_capital(
            load_strategy_states(strategy_states_path(settings.data_dir)),
            evidence_by_slug=load_validation_manifest(validation_manifest_path(settings.data_dir)),
        )
    except Exception:
        allocation = {slug: 1.0 / len(enabled) for slug in enabled}

    report = RebalanceReport(
        asof=asof,
        equity=account.equity,
        enabled_strategies=enabled,
        dry_run=dry_run,
        safety_checks=safety_results,
        halted_strategies=halted,
    )

    all_trade_rows: list[dict[str, object]] = []
    intended: list[OrderTemplate] = []

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
            strategy_equity = account.equity * allocation.get(slug, 0.0)
            target = strategy.target_positions(asof, strategy_equity)
        except Exception as exc:
            logger.exception("strategy {} target_positions raised", slug)
            target = {}
            err: str | None = f"target_positions raised: {exc!r}"
        else:
            err = None

        previous = last_strategy_positions(settings.data_dir, slug)
        orders = reconcile(target=target, current=previous, strategy_slug=slug)
        reference_prices = _latest_reference_prices(
            bars,
            set(target) | set(previous) | {order.symbol for order in orders},
        )

        # Collect — do NOT submit inline. Net submission happens after both loops.
        intended.extend(orders)

        report.outcomes.append(
            StrategyRebalanceOutcome(
                slug=slug,
                target=target,
                previous=previous,
                orders=orders,
                reference_prices=reference_prices,
                error=err,
            )
        )

        # Update our per-strategy bookkeeping with the new target, even in dry-run.
        # Dry-run snapshots are useful in tests but in production daily-rebalance.yml
        # we only commit when dry_run=False.
        if target and not dry_run and record_bookkeeping:
            write_strategy_positions(settings.data_dir, asof, slug, target)

    # Orphan wind-down: exit-only, ADV-capped, fail-closed. Reduces positions of
    # non-live strategies toward flat; never opens. Runs after the live loop.
    for slug in orphans:
        # NOTE: wind-down intentionally runs even if the risk circuit breaker
        # halted live strategies — it is exit-only (reduces exposure), which is
        # the correct action during a drawdown. Orphans are never in `halted`.
        if slug not in REGISTRY:
            report.winddown_outcomes.append(
                WindDownOutcome(slug=slug, error="not registered; manual exit required")
            )
            continue
        try:
            wd_bars = _bars_for(REGISTRY[slug], asof, history_days)
        except Exception as exc:
            report.winddown_outcomes.append(
                WindDownOutcome(slug=slug, error=f"bar fetch failed: {exc!r}")
            )
            continue
        snapshot = last_strategy_positions(settings.data_dir, slug)
        if not any(q != 0 for q in snapshot.values()):
            continue
        result = winddown_orders(
            slug=slug,
            snapshot=snapshot,
            bars=wd_bars,
            asof=asof,
            participation_fraction=winddown_participation,
        )
        # Collect — do NOT submit inline. Net submission happens after both loops.
        # Snapshot is the INTENT (result.remaining) so reconciliation stays
        # consistent; netting removes the opposing-order fail-safe that the old
        # per-exit `persisted` calculation provided.
        intended.extend(result.orders)
        if not dry_run and record_bookkeeping:
            write_strategy_positions(settings.data_dir, asof, slug, result.remaining)
        report.winddown_outcomes.append(
            WindDownOutcome(
                slug=slug,
                exited={o.symbol: o.qty for o in result.orders},
                remaining=result.remaining,
                skipped=result.skipped,
            )
        )

    if not dry_run and intended and record_bookkeeping:
        from datetime import UTC, datetime

        from quant.deploy.markers import write_marker

        # Pre-submit marker: written the instant submission begins, BEFORE the
        # commit/push steps, so a post-submit crash blocks a re-fire next tick.
        write_marker(
            settings.data_dir,
            "daily-rebalance",
            asof,
            kind="SUBMITTED",
            fired_at_utc=datetime.now(UTC),
            exit_code=0,
            duration_s=0.0,
        )

    # Net all intended orders per symbol, then submit once per symbol. This is
    # the single shared account: netting prevents opposing live-vs-orphan orders
    # from being rejected by the broker (wash-trade). Per-strategy snapshots above
    # already recorded intent, so reconciliation stays consistent.
    netted = net_orders(intended)

    # Guard 4: pre-trade portfolio risk gate on the NETTED orders, evaluated
    # against authoritative account equity (gross-exposure + per-symbol
    # concentration). Computed always for observability; a violation REFUSES the
    # entire live batch (fail-closed), while a dry-run records it without blocking.
    # Mandatory hard gate before the live cutover.
    from quant.risk.pretrade import build_pretrade_report

    combined_reference_prices: dict[str, float] = {}
    for outcome in report.outcomes:
        combined_reference_prices.update(outcome.reference_prices)
    pretrade = build_pretrade_report(
        equity=account.equity, orders=netted, reference_prices=combined_reference_prices
    )
    if pretrade.passed:
        safety_results.append(
            CheckResult(
                ok=True,
                name="pretrade_risk",
                detail=f"gross {pretrade.gross_exposure:.2%}, {len(netted)} net orders",
            )
        )
    else:
        violation_detail = "; ".join(v.detail for v in pretrade.violations)
        safety_results.append(CheckResult(ok=False, name="pretrade_risk", detail=violation_detail))
        if not dry_run:
            logger.error(
                "safety: pre-trade risk violation — refusing to submit. {}", violation_detail
            )
            report.skipped_reason = report.skipped_reason or f"pretrade_risk: {violation_detail}"
            netted = []  # fail-closed: refuse the entire batch
        else:
            logger.warning("pre-trade risk violation (dry-run, not blocking): {}", violation_detail)

    # Guard 5: portfolio-level distributional risk gate (WARN-only, fail-OPEN).
    # An independent SECOND guard alongside fail-closed Guard 4. It reconstructs
    # post-trade holdings (broker book + signed netted deltas), characterizes their
    # risk (VaR/CVaR/vol/beta/asset-class), and records a CheckResult + a per-run
    # artifact. In WARN mode (the default) it mutates NOTHING and never blocks — the
    # submit loop below sees byte-identical `netted`. The whole body is wrapped so
    # any failure logs and continues: a bug here can NEVER clear `netted` or abort.
    try:
        from quant.risk.portfolio import (
            PortfolioRisk,
            PortfolioRiskLimits,
            RiskGateMode,
            build_portfolio_risk_gate,
            live_portfolio_risk,
        )

        try:
            gate_mode = RiskGateMode(str(settings.portfolio_risk_gate_mode).strip().lower())
        except ValueError:
            gate_mode = RiskGateMode.WARN  # unknown value -> safe default

        # Post-trade shares = current broker book + signed netted deltas. `netted`
        # is read-only here.
        post_trade: dict[str, int] = {}
        for pos in alpaca_positions:
            qty = int(pos.qty)
            post_trade[pos.symbol] = -qty if str(pos.side) == "short" else qty
        for order in netted:
            delta = order.qty if order.side is OrderSide.BUY else -order.qty
            post_trade[order.symbol] = post_trade.get(order.symbol, 0) + delta
        post_trade = {sym: qty for sym, qty in post_trade.items() if qty != 0}

        port_risk = live_portfolio_risk(post_trade, account.equity, asof=asof)
        if port_risk is None:  # degraded placeholder so the gate still records
            port_risk = PortfolioRisk(
                n_positions=len(post_trade),
                gross_exposure=0.0,
                net_exposure=0.0,
                ann_vol=None,
                var_95=None,
                cvar_95=None,
                beta_to_benchmark=None,
                top_name_weight=None,
                lookback_days=0,
                computable=False,
                degraded_metrics=("ann_vol", "var_95", "cvar_95", "beta_to_benchmark"),
            )
        gate = build_portfolio_risk_gate(port_risk, limits=PortfolioRiskLimits(), mode=gate_mode)
        safety_results.append(
            CheckResult(ok=gate.ok, name="portfolio_risk_gate", detail=gate.detail)
        )
        _write_portfolio_risk_gate_artifact(settings.data_dir, asof=asof, gate=gate)

        if gate_mode is RiskGateMode.BLOCK and not gate.ok and not dry_run:
            # Human-gated BLOCK flip (NOT the default). Mirrors Guard 4: skip the
            # batch only, never halt/de-authorize.
            logger.error("safety: portfolio_risk_gate BLOCK — refusing batch. {}", gate.detail)
            report.skipped_reason = report.skipped_reason or f"portfolio_risk_gate: {gate.detail}"
            netted = []
        elif not gate.ok:
            logger.warning("portfolio_risk_gate WARN (not blocking): {}", gate.detail)
    except Exception:
        logger.exception("Guard 5 (portfolio_risk_gate) failed — continuing (fail-open)")

    submit_failures: list[str] = []
    for order in netted:
        try:
            coid = client.submit_order(order, asof=asof, dry_run=dry_run)
        except Exception:
            # Never silent: log AND record the dropped order so the failure is
            # visible to the report (and the next reconciliation) instead of
            # quietly leaving the book under-positioned.
            logger.exception("net submit_order failed for {}", order.symbol)
            submit_failures.append(order.symbol)
            continue
        all_trade_rows.append(
            {
                "date": pd.Timestamp(asof),
                "strategy": order.strategy_slug,
                "symbol": order.symbol,
                "side": str(order.side),
                "qty": int(order.qty),
                "client_order_id": coid,
                "dry_run": bool(dry_run),
            }
        )

    if submit_failures:
        safety_results.append(
            CheckResult(
                ok=False,
                name="submit_failures",
                detail=(
                    f"{len(submit_failures)} order(s) failed to submit: "
                    f"{', '.join(submit_failures)}"
                ),
            )
        )

    if all_trade_rows and not dry_run and record_bookkeeping:
        append_trades(settings.data_dir, all_trade_rows)

    return report
