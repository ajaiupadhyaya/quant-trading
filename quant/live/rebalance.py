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

import numpy as np
import pandas as pd

from quant.backtest.impact import trailing_dollar_adv
from quant.data.bars import BarRequest, get_bars
from quant.data.universe import SLEEVE_UNIVERSE
from quant.execution.alpaca import AlpacaClient
from quant.execution.netting import net_orders
from quant.execution.orders import OrderSide, OrderTemplate
from quant.execution.policy import ExecutionPolicyConfig, apply_execution_policy
from quant.execution.reconciler import reconcile
from quant.governance.allocation import (
    AllocationConfig,
    allocate_capital,
    load_strategy_returns,
)
from quant.live.bookkeeping import (
    append_equity_row,
    append_trades,
    last_strategy_positions,
    write_strategy_positions,
)
from quant.live.derisk import (
    DeriskConfig,
    derisk_multiplier,
    load_engine_state,
    to_report_dict,
)
from quant.live.voltarget import (
    VolTargetConfig,
    voltarget_multiplier,
)
from quant.live.voltarget import to_report_dict as voltarget_to_report_dict
from quant.strategies import REGISTRY
from quant.strategies.base import Strategy
from quant.util.config import Settings
from quant.util.logging import logger


def exclude_sleeve_positions(post_trade: dict[str, int]) -> dict[str, int]:
    """Drop intraday-sleeve symbols from a position book.

    The intraday live loop trades a disjoint sleeve (``SLEEVE_UNIVERSE``) inside the
    same Alpaca account. Those holdings belong to the sleeve, not the daily system,
    so they must be excluded from the daily portfolio-risk gate's view — otherwise a
    sleeve position would inflate the daily gross/vol/beta and could trip Guard-5.
    """
    return {sym: qty for sym, qty in post_trade.items() if sym not in SLEEVE_UNIVERSE}


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
    derisk: dict[str, Any] | None = None  # deterministic de-risk overlay (shadow unless actuated)
    voltarget: dict[str, Any] | None = None  # forecast-vol-target overlay (shadow unless actuated)

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


def _dollar_adv_for(bars: pd.DataFrame, symbols: set[str], asof: date) -> dict[str, float]:
    """Trailing dollar-ADV per symbol from a strategy's bars (PIT: strictly-prior).

    Reuses the backtest impact helper so live participation control and backtest
    impact accounting share one definition. ``window`` is fixed at 21 to match
    ``backtest.impact``'s calibration; a symbol with no estimable ADV is omitted
    (the policy treats a missing key as fail-open passthrough).
    """
    fill_ts = pd.Timestamp(asof)
    adv: dict[str, float] = {}
    for symbol in sorted(symbols):
        value = trailing_dollar_adv(bars, symbol, fill_ts, window=21)
        if value > 0.0:
            adv[symbol] = value
    return adv


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


def _write_portfolio_risk_gate_artifact(
    data_dir: Path, *, asof: date, gate: Any, stress: Any | None = None
) -> None:
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
    if stress is not None:
        payload["stress"] = {
            "computable": stress.computable,
            "worst_loss": stress.worst_loss,
            "worst_scenario": stress.worst_scenario,
            "degraded": list(stress.degraded),
            "results": [
                {
                    "name": res.name,
                    "kind": res.kind,
                    "pnl_pct": res.pnl_pct,
                    "by_class": dict(res.by_class),
                    "missing_symbols": list(res.missing_symbols),
                    "computable": res.computable,
                }
                for res in stress.results
            ],
        }
    path = data_dir / "risk" / f"portfolio_risk_gate.{asof.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def _write_execution_plan_artifact(
    data_dir: Path, *, asof: date, rows: list[dict[str, Any]]
) -> None:
    """Write the impact-aware execution plan (atomic JSON). One row per order the
    policy considered: original/capped/deferred qty, participation, and the
    resulting order_type/limit_price. Observability for the participation caps and
    marketable-limit re-pricing the live executor applied."""
    from quant.util.atomic import write_json_atomic

    payload = {"asof": asof.isoformat(), "orders": rows}
    path = data_dir / "live" / f"execution_plan.{asof.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def _write_allocation_compare_artifact(
    data_dir: Path,
    *,
    asof: date,
    states: Any,
    evidence: Any,
    returns_by_slug: Any,
    active: Any,
) -> None:
    """Write an observed equal-live-vs-risk-based allocation comparison (atomic JSON).

    Shows what each mode WOULD allocate (and per-strategy mean/std) so a risk-based
    split can be evaluated before it is consciously enabled. Observability only —
    it does not change the actuated allocation (the caller already computed that)."""
    from dataclasses import replace

    from quant.governance.allocation import allocate_capital, strategy_risk
    from quant.governance.models import GovernanceState
    from quant.util.atomic import write_json_atomic

    def _weights(mode: str) -> dict[str, float]:
        return allocate_capital(
            states,
            evidence_by_slug=evidence,
            config=replace(active, mode=mode),
            returns_by_slug=returns_by_slug,
        )

    live_slugs = sorted(slug for slug, st in states.items() if st.state is GovernanceState.LIVE)
    risk: dict[str, dict[str, float | None]] = {}
    for slug in live_slugs:
        r = returns_by_slug.get(slug)
        if r is None:
            risk[slug] = {"mean": None, "std": None, "n_obs": 0}
            continue
        mean, std = strategy_risk(r, active.min_observations)
        risk[slug] = {
            "mean": None if mean != mean else mean,  # NaN -> None
            "std": None if std != std else std,
            "n_obs": len(r),
        }

    payload = {
        "asof": asof.isoformat(),
        "active_mode": active.mode,
        "active_weights": _weights(active.mode),
        "equal_live_weights": _weights("equal-live"),
        "risk_parity_weights": _weights("risk-parity"),
        "fractional_kelly_weights": _weights("fractional-kelly"),
        "hrp_weights": _weights("hrp"),
        "per_strategy_risk": risk,
    }
    path = data_dir / "governance" / f"allocation_compare.{asof.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def _book_returns_for_voltarget(settings: Settings, allocation: dict[str, float]) -> np.ndarray:
    """Allocation-weighted blend of the live strategies' OOS curves — the book's vol proxy.

    The live equity curve is far too short (days) to fit a vol forecast, so the
    forecast-vol-target overlay reads the book's *representative* return history:
    each live strategy's walk-forward OOS daily returns, date-aligned and blended
    at the current allocation weights. Best-effort — any unreadable curve is
    dropped; an empty result makes the overlay degrade to a no-op (fail-safe).
    """
    import pandas as pd

    cols: dict[str, pd.Series] = {}
    for slug, weight in allocation.items():
        if weight <= 0.0:
            continue
        path = settings.data_dir / "backtests" / slug / "walkforward.parquet"
        if not path.exists():
            continue
        try:
            equity = pd.read_parquet(path)["equity"].astype(float)
            cols[slug] = equity.pct_change().dropna()
        except Exception:
            continue
    if not cols:
        return np.array([], dtype=float)
    frame = pd.DataFrame(cols).dropna(how="any")
    if frame.empty:
        return np.array([], dtype=float)
    weights = np.array([allocation[s] for s in frame.columns], dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    return np.asarray(frame.to_numpy(dtype=float) @ weights, dtype=float)


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
    exec_policy: ExecutionPolicyConfig | None = None,
    alloc_config: AllocationConfig | None = None,
    derisk_config: DeriskConfig | None = None,
    voltarget_config: VolTargetConfig | None = None,
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

    from quant.governance.store import (
        load_strategy_states,
        load_validation_manifest,
        strategy_states_path,
        validation_manifest_path,
    )

    alloc_config = alloc_config or AllocationConfig()
    try:
        states = load_strategy_states(strategy_states_path(settings.data_dir))
        evidence = load_validation_manifest(validation_manifest_path(settings.data_dir))
        # Risk-based modes need per-strategy OOS return curves; equal-live/evidence
        # modes ignore them. Loading is best-effort (missing curve -> fail-open).
        returns_by_slug = load_strategy_returns(evidence, root=settings.data_dir.parent)
        allocation = allocate_capital(
            states,
            evidence_by_slug=evidence,
            config=alloc_config,
            returns_by_slug=returns_by_slug,
        )
    except Exception:
        logger.exception("capital allocation failed — falling back to equal-split")
        allocation = {slug: 1.0 / len(enabled) for slug in enabled}
    else:
        # Observed-first comparison (equal-live vs risk-based). Best-effort and
        # fully isolated: a failure here can NEVER alter the actuated allocation.
        try:
            _write_allocation_compare_artifact(
                settings.data_dir,
                asof=asof,
                states=states,
                evidence=evidence,
                returns_by_slug=returns_by_slug,
                active=alloc_config,
            )
        except Exception:
            logger.exception("allocation compare artifact failed (non-fatal)")

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
    # Per-symbol trailing dollar-ADV, accumulated from each strategy's bars, used
    # by the impact-aware execution policy after netting. Disabled by default
    # (exec_policy=None ⇒ identity), so this is observability-only until enabled.
    combined_dollar_adv: dict[str, float] = {}
    exec_policy = exec_policy or ExecutionPolicyConfig()

    # Deterministic one-way de-risk overlay from the continuous engine's MarketState.
    # Default SHADOW (actuate=False ⇒ applied=1.0): computed + reported, changes nothing.
    # When actuated, it can only SHRINK each strategy's equity slice (>= floor), and the
    # next rebalance restores full size — all within the existing halt + Guard 4/5 envelope.
    derisk_config = derisk_config or DeriskConfig()
    derisk = derisk_multiplier(load_engine_state(settings.data_dir), derisk_config)
    report.derisk = to_report_dict(derisk)
    if derisk.reasons and not derisk.degraded:
        verb = "APPLYING" if derisk.actuated else "shadow"
        logger.info(
            "derisk overlay {}: computed x{} applied x{} ({})",
            verb,
            derisk.multiplier,
            derisk.applied,
            ", ".join(derisk.reasons),
        )

    # Forecast-vol-target overlay (gate-passed, default SHADOW). Reads the book's
    # representative return history (allocation-blended OOS curves) and cuts gross
    # one-way when the validated vol forecast exceeds the book's trailing vol. Like
    # de-risk it can ONLY shrink gross; both compose multiplicatively below.
    voltarget_config = voltarget_config or VolTargetConfig()
    voltarget = voltarget_multiplier(
        _book_returns_for_voltarget(settings, allocation), voltarget_config
    )
    report.voltarget = voltarget_to_report_dict(voltarget)
    if not voltarget.degraded and voltarget.reasons:
        verb = "APPLYING" if voltarget.actuated else "shadow"
        logger.info(
            "voltarget overlay {}: computed x{} applied x{} ({})",
            verb,
            voltarget.multiplier,
            voltarget.applied,
            ", ".join(voltarget.reasons),
        )

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
            # Both overlays' `applied` are 1.0 in shadow mode (byte-identical) and a
            # one-way factor (<= 1.0) only when actuation is consciously enabled; they
            # compose multiplicatively, so the book can only ever be de-risked.
            strategy_equity = (
                account.equity * allocation.get(slug, 0.0) * derisk.applied * voltarget.applied
            )
            target = strategy.target_positions(asof, strategy_equity)
        except Exception as exc:
            logger.exception("strategy {} target_positions raised", slug)
            target = {}
            err: str | None = f"target_positions raised: {exc!r}"
        else:
            err = None

        previous = last_strategy_positions(settings.data_dir, slug)
        orders = reconcile(target=target, current=previous, strategy_slug=slug)
        order_symbols = set(target) | set(previous) | {order.symbol for order in orders}
        reference_prices = _latest_reference_prices(bars, order_symbols)
        combined_dollar_adv.update(_dollar_adv_for(bars, order_symbols, asof))

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

    combined_reference_prices: dict[str, float] = {}
    for outcome in report.outcomes:
        combined_reference_prices.update(outcome.reference_prices)

    # Impact-aware execution policy: cap each netted order to a max participation
    # of trailing dollar-ADV (residual carried to the next session by reconcile)
    # and optionally re-price high-participation orders as marketable limits. The
    # risk gates below then evaluate exactly what will be submitted. Disabled by
    # default (exec_policy=None ⇒ identity), so the netted batch is unchanged
    # until consciously enabled.
    netted, exec_plan_rows = apply_execution_policy(
        netted,
        dollar_adv=combined_dollar_adv,
        reference_prices=combined_reference_prices,
        cfg=exec_policy,
    )
    if exec_plan_rows:
        _write_execution_plan_artifact(settings.data_dir, asof=asof, rows=exec_plan_rows)

    # Guard 4: pre-trade portfolio risk gate on the NETTED orders, evaluated
    # against authoritative account equity (gross-exposure + per-symbol
    # concentration). Computed always for observability; a violation REFUSES the
    # entire live batch (fail-closed), while a dry-run records it without blocking.
    # Mandatory hard gate before the live cutover.
    from quant.risk.pretrade import build_pretrade_report

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
        from quant.risk.scenarios import live_stress

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
        # Exclude intraday-sleeve holdings: they share this Alpaca account but belong
        # to the intraday loop, not the daily system, and must not enter Guard-5.
        post_trade = exclude_sleeve_positions(post_trade)

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
        stress = live_stress(post_trade, account.equity, asof=asof)
        gate = build_portfolio_risk_gate(
            port_risk, limits=PortfolioRiskLimits(), mode=gate_mode, stress=stress
        )
        safety_results.append(
            CheckResult(ok=gate.ok, name="portfolio_risk_gate", detail=gate.detail)
        )
        _write_portfolio_risk_gate_artifact(settings.data_dir, asof=asof, gate=gate, stress=stress)

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
