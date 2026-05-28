"""Monitoring daemon orchestration: gather inputs, run a tick, write status.

The daemon can HALT but NEVER resumes — resume is always a manual
`quant governance resume`. Its only side effects are ``set_halt`` (the
kill-switch) and the status artifact; it never touches orders. Fail-safe:
on missing/empty equity the inputs evaluate to ``ok`` so a monitoring gap
cannot trigger a false halt.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from quant.execution.alpaca import PositionRow
from quant.governance.drift import DriftRow, summarize_drift
from quant.governance.halt import load_halt, set_halt
from quant.live.bookkeeping import read_equity
from quant.live.safety import (
    CheckResult,
    check_bar_freshness,
    check_reconciliation,
    enabled_strategy_slugs,
)
from quant.monitor.guardrails import (
    GuardrailConfig,
    GuardrailInputs,
    GuardrailReport,
    evaluate_guardrails,
)
from quant.monitor.status import MonitorStatus, write_status
from quant.util.logging import logger


@dataclass(frozen=True)
class TickResult:
    report: GuardrailReport
    halt_triggered: bool  # set_halt was called THIS tick
    halt_active: bool  # halt active after this tick
    heartbeat: str
    at: datetime


def _drift_rows(equity_df: pd.DataFrame, config: GuardrailConfig) -> list[DriftRow]:
    if equity_df.empty or "equity" not in equity_df.columns:
        return []
    returns = equity_df["equity"].astype(float).pct_change(fill_method=None).dropna()
    if returns.empty:
        return []
    realized = {"account": returns}
    expected = {"account": pd.Series(0.0, index=returns.index)}
    return summarize_drift(realized, expected, config=config.drift)


def _account_drawdown(equity_df: pd.DataFrame, lookback_days: int) -> float:
    """Worst trailing peak-to-trough drawdown. Mirrors safety._recent_drawdown_pct."""
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    window = equity_df.tail(lookback_days)
    if window.empty:
        return 0.0
    equity = window["equity"].astype(float)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _latest_equity(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    return float(equity_df["equity"].astype(float).iloc[-1])


def gather_inputs(
    data_dir: Path,
    *,
    asof: date,
    config: GuardrailConfig,
    alpaca_positions: list[PositionRow] | None = None,
    symbols: list[str] | None = None,
) -> GuardrailInputs:
    """Read local state into a GuardrailInputs. I/O side of the daemon.

    Reconciliation is included only when ``alpaca_positions`` is provided;
    bar-freshness only when ``symbols`` is non-empty. Both default to skipped.
    """
    equity_df = read_equity(data_dir)
    drift_rows = _drift_rows(equity_df, config)
    dd = _account_drawdown(equity_df, config.risk.drawdown_lookback_days)

    reconciliation: CheckResult | None = None
    if alpaca_positions is not None:
        reconciliation = check_reconciliation(
            data_dir=data_dir,
            alpaca_positions=alpaca_positions,
            enabled_slugs=enabled_strategy_slugs(),
        )

    freshness: CheckResult | None = None
    if symbols:
        freshness = check_bar_freshness(data_dir, symbols=symbols, asof=asof)

    return GuardrailInputs(
        drift_rows=drift_rows,
        account_drawdown_pct=dd,
        latest_equity=_latest_equity(equity_df),
        reconciliation=reconciliation,
        bar_freshness=freshness,
    )


def format_heartbeat(
    inputs: GuardrailInputs,
    report: GuardrailReport,
    now: datetime,
    *,
    halt_active: bool,
) -> str:
    ts = now.strftime("%H:%M:%S")
    eq = f"${inputs.latest_equity:,.0f}"
    dd = f"{inputs.account_drawdown_pct:.1%}"
    parts = " | ".join(f"{o.name} {o.severity}" for o in report.outcomes)
    line = f"{ts} | equity {eq} dd {dd} | {parts}"
    if halt_active:
        line += " | [HALT]"
    return line


def run_once(
    data_dir: Path,
    config: GuardrailConfig,
    *,
    asof: date | None = None,
    inputs: GuardrailInputs | None = None,
    alpaca_positions: list[PositionRow] | None = None,
    symbols: list[str] | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> TickResult:
    """One monitoring tick. Evaluate guardrails; auto-halt on a halt verdict
    (unless already halted or dry_run); always write the status artifact."""
    asof = asof or date.today()
    now = now or datetime.now(UTC).replace(microsecond=0)
    if inputs is None:
        inputs = gather_inputs(
            data_dir,
            asof=asof,
            config=config,
            alpaca_positions=alpaca_positions,
            symbols=symbols,
        )

    report = evaluate_guardrails(inputs, config)
    halt_active = load_halt(data_dir).active
    triggered = False
    if report.halting and not halt_active and not dry_run:
        names = ",".join(o.name for o in report.outcomes if o.severity == "halt")
        set_halt(data_dir, reason=f"auto-halt: {names}", created_at=now)
        halt_active = True
        triggered = True

    heartbeat = format_heartbeat(inputs, report, now, halt_active=halt_active)
    write_status(
        data_dir,
        MonitorStatus(
            version=1,
            at=now,
            worst_severity=report.worst_severity,
            halt_triggered_this_tick=triggered,
            halt_active=halt_active,
            outcomes=report.outcomes,
            heartbeat=heartbeat,
        ),
    )
    return TickResult(
        report=report,
        halt_triggered=triggered,
        halt_active=halt_active,
        heartbeat=heartbeat,
        at=now,
    )


def run_loop(
    data_dir: Path,
    config: GuardrailConfig,
    *,
    interval_s: float = 300.0,
    dry_run: bool = False,
    max_ticks: int | None = None,
    inputs_fn: Callable[[], GuardrailInputs] | None = None,
    alpaca_positions_fn: Callable[[], list[PositionRow] | None] | None = None,
    symbols: list[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    console_print: Callable[[str], None] | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> list[TickResult]:
    """Repeatedly run a tick, printing the heartbeat, sleeping between ticks.

    Fail-safe: a tick that raises is caught, logged (and reported via
    ``console_print`` if given), and the loop continues. ``inputs_fn`` (test
    seam) supplies inputs directly; otherwise ``alpaca_positions_fn`` is
    consulted each tick for reconciliation. Stops after ``max_ticks`` ticks
    (None = forever).
    """
    results: list[TickResult] = []
    tick = 0
    while max_ticks is None or tick < max_ticks:
        try:
            tick_inputs = inputs_fn() if inputs_fn is not None else None
            positions = (
                alpaca_positions_fn()
                if (tick_inputs is None and alpaca_positions_fn is not None)
                else None
            )
            now = now_fn() if now_fn is not None else None
            res = run_once(
                data_dir,
                config,
                inputs=tick_inputs,
                alpaca_positions=positions,
                symbols=symbols,
                dry_run=dry_run,
                now=now,
            )
            if console_print is not None:
                console_print(res.heartbeat)
            results.append(res)
        except Exception as exc:  # fail-safe: never crash the loop
            # Always log so a headless daemon (no console sink) never fails silently.
            logger.warning("monitor tick error (continuing): {!r}", exc)
            if console_print is not None:
                console_print(f"monitor tick error (continuing): {exc!r}")
        tick += 1
        if max_ticks is not None and tick >= max_ticks:
            break
        sleep(interval_s)
    return results
