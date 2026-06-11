"""Click CLI: top-level group + every subcommand wired to the strategy registry.

Foundation phase: most subcommands are stubs that raise `click.ClickException`
with a clear "not yet implemented in Plan N" message. `status` and `data` are
fully functional; the rest are scaffolded so the command surface is stable.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
import webbrowser
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from quant.backtest import BacktestConfig, run_walkforward, write_tearsheet
from quant.data.bars import BarRequest, get_bars
from quant.data.refresh import refresh_caches
from quant.execution.alpaca import AlpacaClient
from quant.strategies import REGISTRY, list_strategies
from quant.util.config import Settings
from quant.util.logging import configure_logging, logger

console = Console()


def _require_strategy(slug: str) -> str:
    if slug not in REGISTRY:
        known = ", ".join(s.slug for s in list_strategies()) or "(none registered)"
        raise click.ClickException(f"Unknown strategy {slug!r}. Known: {known}")
    return slug


@click.group(name="quant", help="Systematic trading: backtest, validate, rebalance, monitor.")
@click.option("--log-level", default=None, help="Override log level (DEBUG/INFO/WARNING).")
def cli(log_level: str | None) -> None:
    settings = Settings.model_construct() if not _can_load_settings() else Settings()  # type: ignore[call-arg]
    level: str = log_level or str(getattr(settings, "log_level", "INFO"))
    configure_logging(level)


def _can_load_settings() -> bool:
    try:
        Settings()  # type: ignore[call-arg]
        return True
    except Exception:  # CLI help path must not require env
        return False


# --- intraday sub-system ----------------------------------------------------
from quant.intraday.cli import intraday as _intraday_group  # noqa: E402

cli.add_command(_intraday_group)
# ----------------------------------------------------------------------------


def _doctor_governance_live_slugs(data_dir: Path) -> tuple[list[str], str | None]:
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import load_strategy_states, strategy_states_path

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError as exc:
        return [], str(exc)
    live = sorted(
        slug
        for slug, state in states.items()
        if slug in REGISTRY and state.state is GovernanceState.LIVE
    )
    if not live:
        return [], "No governance-live strategies; run `quant governance refresh`."
    return live, None


def _default_data_quality_end_date(today: date) -> date:
    from quant.util.trading_calendar import previous_trading_day

    return previous_trading_day(today)


@cli.command(help="Show Alpaca account snapshot and per-strategy attribution.")
def status() -> None:
    client = AlpacaClient()
    acct = client.account()
    positions = client.positions()

    acct_table = Table(title="Account", show_header=True)
    acct_table.add_column("Field")
    acct_table.add_column("Value", justify="right")
    acct_table.add_row("Equity", f"${acct.equity:,.2f}")
    acct_table.add_row("Last Equity", f"${acct.last_equity:,.2f}")
    acct_table.add_row("Cash", f"${acct.cash:,.2f}")
    acct_table.add_row("Buying Power", f"${acct.buying_power:,.2f}")
    acct_table.add_row("Pattern Day Trader", str(acct.pattern_day_trader))
    console.print(acct_table)

    if positions:
        pos_table = Table(title=f"Positions ({len(positions)})", show_header=True)
        for col in ("Symbol", "Qty", "Avg", "Last", "Mkt Value", "Unrealized PnL"):
            pos_table.add_column(col, justify="right" if col != "Symbol" else "left")
        for p in positions:
            pos_table.add_row(
                p.symbol,
                str(p.qty),
                f"${p.avg_entry_price:,.2f}",
                f"${p.current_price:,.2f}",
                f"${p.market_value:,.2f}",
                f"${p.unrealized_pl:,.2f}",
            )
        console.print(pos_table)
    else:
        console.print("[dim]No open positions.[/dim]")


@cli.command(help="Run full walk-forward backtest for <strategy> and write tear-sheet.")
@click.argument("strategy")
@click.option(
    "--quick",
    is_flag=True,
    help="Skip grid search; use strategy defaults only (much faster).",
)
@click.option(
    "--start",
    default="2010-01-01",
    show_default=True,
    help="History start date (YYYY-MM-DD).",
)
@click.option("--end", default=None, help="History end date (YYYY-MM-DD). Default: today.")
def backtest(strategy: str, quick: bool, start: str, end: str | None) -> None:
    _require_strategy(strategy)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    settings = Settings()  # type: ignore[call-arg]
    strategy_cls = REGISTRY[strategy]
    universe = list(strategy_cls.spec.universe)
    grid: dict[str, Sequence[Any]] = (
        {} if quick else {k: list(v) for k, v in strategy_cls.param_grid.items()}
    )
    if quick:
        logger.info("--quick: skipping grid search; running with strategy defaults only.")
    else:
        n_combos = 1
        for vals in grid.values():
            n_combos *= max(len(vals), 1)
        logger.info(
            "Grid search across {} param combos: {}", n_combos, ", ".join(grid) or "(defaults)"
        )

    console.print(f"[bold]Fetching bars for {len(universe)} symbols...[/bold]")
    bars = get_bars(BarRequest(symbols=universe, start=start_date, end=end_date))
    if bars.empty:
        raise click.ClickException(
            f"No bars returned for {strategy!r} over {start_date}..{end_date}."
        )

    def factory(params: dict[str, object], bars_for_strategy):  # type: ignore[no-untyped-def]
        return strategy_cls.build(bars=bars_for_strategy, params=params)

    console.print("[bold]Running walk-forward...[/bold]")
    result = run_walkforward(
        strategy_factory=factory,
        param_grid=grid,
        bars=bars,
        start=start_date,
        end=end_date,
        config=BacktestConfig(),
    )

    out_dir = settings.data_dir / "backtests" / strategy
    html_path = write_tearsheet(
        result=result,
        slug=strategy,
        strategy_name=strategy_cls.spec.name,
        out_dir=out_dir,
        write_chosen_params=not quick,
    )
    console.print(f"[green]Wrote {html_path}[/green]")


@cli.command(
    "combined-book",
    help="Backtest all live-enabled strategies into one joint equity curve.",
)
@click.option(
    "--start", default="2018-01-01", show_default=True, help="History start (YYYY-MM-DD)."
)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
def combined_book(start: str, end: str | None) -> None:
    from quant.backtest import run_combined_book
    from quant.backtest.activity import annualized_turnover, capacity_report
    from quant.backtest.metrics import cagr, max_drawdown, sharpe

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    enabled = sorted(slug for slug, cls in REGISTRY.items() if cls.spec.enabled_live)
    if not enabled:
        raise click.ClickException("No live-enabled strategies registered.")

    strategies: dict[str, Any] = {}
    bars_per: dict[str, pd.DataFrame] = {}
    for slug in enabled:
        cls = REGISTRY[slug]
        console.print(f"[bold]Fetching bars for {slug} ({len(cls.spec.universe)} symbols)...[/]")
        b = get_bars(BarRequest(symbols=list(cls.spec.universe), start=start_date, end=end_date))
        if b.empty:
            console.print(f"[red]No bars for {slug}; skipping.[/]")
            continue
        strategies[slug] = cls.build(bars=b)
        bars_per[slug] = b

    if not strategies:
        raise click.ClickException("No strategies had bars to run on.")

    cfg = BacktestConfig()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=cfg,
        start=start_date,
        end=end_date,
    )

    def _cap(trades: pd.DataFrame, equity: pd.Series) -> str:
        rep = capacity_report(trades, equity, impact_coef_bps=cfg.impact_coef_bps)
        if rep.binding == "none" or rep.capacity_aum <= 0.0:
            return "n/a"
        tag = "part" if rep.binding == "participation" else "impact"
        aum = rep.capacity_aum
        unit = (
            f"${aum / 1e9:.1f}B"
            if aum >= 1e9
            else f"${aum / 1e6:.1f}M"
            if aum >= 1e6
            else f"${aum / 1e3:.0f}K"
        )
        return f"{unit} ({tag})"

    table = Table(title="Combined-book result", show_header=True)
    table.add_column("Strategy")
    table.add_column("Alloc", justify="right")
    table.add_column("End Equity", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Turnover", justify="right")
    table.add_column("Financing $", justify="right")
    table.add_column("Capacity", justify="right")
    for slug in sorted(result.per_strategy):
        sub = result.per_strategy[slug]
        table.add_row(
            slug,
            f"{result.allocation.get(slug, 0):.1%}",
            f"${sub.ending_equity:,.0f}",
            f"{sharpe(sub.returns):.2f}",
            f"{cagr(sub.returns):.2%}",
            f"{max_drawdown(sub.returns):.2%}",
            f"{annualized_turnover(sub.trades, sub.equity_curve):.0%}",
            f"${float(cast(Any, sub.metadata.get('financing_cost_total')) or 0.0):,.0f}",
            _cap(sub.trades, sub.equity_curve),
        )
    table.add_section()
    combined_financing: float = sum(
        float(cast(Any, s.metadata.get("financing_cost_total")) or 0.0)
        for s in result.per_strategy.values()
    )
    table.add_row(
        "[bold]COMBINED[/]",
        "100.0%",
        f"${result.ending_equity:,.0f}",
        f"{sharpe(result.returns):.2f}",
        f"{cagr(result.returns):.2%}",
        f"{max_drawdown(result.returns):.2%}",
        f"{annualized_turnover(result.trades, result.equity_curve):.0%}",
        f"${combined_financing:,.0f}",
        _cap(result.trades, result.equity_curve),
    )
    console.print(table)

    from quant.backtest import write_combined_tearsheet

    settings = Settings()  # type: ignore[call-arg]
    out_dir = settings.data_dir / "backtests" / "_combined"
    html_path = write_combined_tearsheet(result=result, out_dir=out_dir)
    console.print(f"[green]Wrote {html_path}[/green]")


def _validation_command(
    *,
    strategy: str,
    start_date: date,
    end_date: date,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    quick: bool,
) -> str:
    command = (
        f"quant validate {strategy} --start {start_date} --end {end_date} "
        f"--bootstrap-resamples {bootstrap_resamples} --bootstrap-seed {bootstrap_seed}"
    )
    if quick:
        command += " --quick"
    return command


def _write_validation_report_json(
    *,
    out_dir: Path,
    slug: str,
    run_date: date,
    data_start: date,
    data_end: date,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    validation_command: str,
    report: Any,
    provenance: str,
) -> Path:
    import json

    from quant.backtest.validation import EVIDENCE_SCHEMA_VERSION

    n_tested = sum(1 for r in report.regime_breakdown if r.n_days >= 30)
    payload = {
        "evidence_schema_version": int(EVIDENCE_SCHEMA_VERSION),
        "slug": slug,
        "run_date": run_date.isoformat(),
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "gate_deflated_sharpe": bool(report.gate_deflated_sharpe),
        "gate_probabilistic_sharpe": bool(report.gate_probabilistic_sharpe),
        "gate_bootstrap_lower": bool(report.gate_bootstrap_lower),
        "gate_regime": bool(report.gate_regime),
        "gate_holdout": bool(report.gate_holdout),
        "deflated_sharpe": float(report.deflated_sharpe),
        "probabilistic_sharpe": float(report.probabilistic_sharpe),
        "bootstrap_total_return_p05": (
            None if report.bootstrap_ci is None else float(report.bootstrap_ci.total_return_p05)
        ),
        "n_positive_regimes": int(report.n_positive_regimes),
        "n_tested_regimes": int(n_tested),
        "holdout_total_return": (
            None if report.holdout is None else float(report.holdout.total_return)
        ),
        "validation_command": validation_command,
        "provenance": provenance,
    }
    path = out_dir / "validation_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@cli.command(help="Run the full validation battery (walk-forward + CPCV + DSR + ...).")
@click.argument("strategy")
@click.option(
    "--start", default="2010-01-01", show_default=True, help="History start date (YYYY-MM-DD)."
)
@click.option("--end", default=None, help="History end date (YYYY-MM-DD). Default: today.")
@click.option("--bootstrap-resamples", default=1000, show_default=True, type=int)
@click.option("--bootstrap-seed", default=0, show_default=True, type=int)
@click.option("--cpcv-groups", default=6, show_default=True, type=int)
@click.option("--cpcv-k-test", default=2, show_default=True, type=int)
@click.option("--quick", is_flag=True, help="Skip grid search; use strategy defaults only.")
@click.option(
    "--holdout-years",
    default=1,
    show_default=True,
    type=int,
    help="Reserve trailing N years as never-seen holdout test (0 = disabled).",
)
def validate(
    strategy: str,
    start: str,
    end: str | None,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    cpcv_groups: int,
    cpcv_k_test: int,
    quick: bool,
    holdout_years: int,
) -> None:
    from datetime import timedelta as _td

    from quant.backtest.cpcv import CPCVConfig
    from quant.backtest.validation import run_validation
    from quant.data.snapshot import create_data_snapshot
    from quant.research.registry import ExperimentRecord, append_experiment

    _require_strategy(strategy)
    started = time.monotonic()

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    # Reserve trailing `holdout_years` as never-seen holdout.
    wf_end = end_date
    holdout_start: date | None = None
    holdout_end: date | None = None
    if holdout_years > 0:
        holdout_end = end_date
        holdout_start = end_date.replace(year=end_date.year - holdout_years) + _td(days=1)
        wf_end = holdout_start - _td(days=1)
        if wf_end <= start_date:
            raise click.ClickException(
                f"holdout-years={holdout_years} leaves no walk-forward window "
                f"({start_date}..{wf_end}); shrink the holdout or extend --start."
            )

    settings = Settings()  # type: ignore[call-arg]
    strategy_cls = REGISTRY[strategy]
    universe = list(strategy_cls.spec.universe)
    grid: dict[str, Sequence[Any]] = (
        {} if quick else {k: list(v) for k, v in strategy_cls.param_grid.items()}
    )

    console.print(f"[bold]Fetching bars for {len(universe)} symbols...[/bold]")
    bars = get_bars(BarRequest(symbols=universe, start=start_date, end=end_date))
    if bars.empty:
        raise click.ClickException(
            f"No bars returned for {strategy!r} over {start_date}..{end_date}."
        )
    snapshot = create_data_snapshot(
        settings.data_dir,
        symbols=universe,
        start=start_date,
        end=end_date,
    )

    def factory(params: dict[str, object], bars_for_strategy):  # type: ignore[no-untyped-def]
        return strategy_cls.build(bars=bars_for_strategy, params=params)

    console.print(
        f"[bold]Running walk-forward over {start_date}..{wf_end} "
        f"(holdout {holdout_start}..{holdout_end})...[/bold]"
    )
    wf = run_walkforward(
        strategy_factory=factory,
        param_grid=grid,
        bars=bars,
        start=start_date,
        end=wf_end,
        config=BacktestConfig(),
    )
    chosen = wf.per_window_params[-1][1] if wf.per_window_params else {}

    console.print("[bold]Running validation battery (CPCV + DSR + bootstrap + regimes)...[/bold]")
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=factory,
        chosen_params=chosen,
        backtest_config=BacktestConfig(),
        cpcv_config=CPCVConfig(n_groups=cpcv_groups, k_test=cpcv_k_test),
        bootstrap_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )

    out_dir = settings.data_dir / "backtests" / strategy
    html_path = write_tearsheet(
        result=wf,
        slug=strategy,
        strategy_name=strategy_cls.spec.name,
        out_dir=out_dir,
        validation=report,
        write_chosen_params=not quick,
    )
    validation_json = _write_validation_report_json(
        out_dir=out_dir,
        slug=strategy,
        run_date=date.today(),
        data_start=start_date,
        data_end=end_date,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        validation_command=_validation_command(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            quick=quick,
        ),
        report=report,
        provenance=f"quant validate {strategy} --start {start_date} --end {end_date}",
    )
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = "unknown"
    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"{strategy}-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy=strategy,
            kind="validation",
            git_sha=git_sha,
            command=_validation_command(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                quick=quick,
            ),
            params=dict(chosen),
            metrics={
                "dsr": float(report.deflated_sharpe),
                "psr": float(report.probabilistic_sharpe),
                "bootstrap_lower_5": float(
                    report.bootstrap_ci.total_return_p05 if report.bootstrap_ci else 0.0
                ),
                "holdout_total_return": float(
                    report.holdout.total_return if report.holdout is not None else 0.0
                ),
            },
            gates={
                "deflated_sharpe": bool(report.gate_deflated_sharpe),
                "probabilistic_sharpe": bool(report.gate_probabilistic_sharpe),
                "bootstrap_lower": bool(report.gate_bootstrap_lower),
                "regime": bool(report.gate_regime),
                "holdout": bool(report.gate_holdout),
                "overall": bool(report.passed),
            },
            artifacts={
                "tearsheet": str(html_path),
                "validation_report": str(validation_json),
                "walkforward": str(out_dir / "walkforward.parquet"),
            },
            data_snapshot_id=snapshot.snapshot_id,
            wall_time_seconds=round(time.monotonic() - started, 3),
        ),
    )

    table = Table(title=f"Validation report — {strategy}")
    table.add_column("Gate")
    table.add_column("Value")
    table.add_column("Threshold")
    table.add_column("Pass?")
    table.add_row(
        "Deflated Sharpe",
        f"{report.deflated_sharpe:.3f}",
        "≥ 0.30",
        "✓" if report.gate_deflated_sharpe else "✗",
    )
    table.add_row(
        "Probabilistic Sharpe",
        f"{report.probabilistic_sharpe:.3f}",
        "≥ 0.70",
        "✓" if report.gate_probabilistic_sharpe else "✗",
    )
    boot_lower = (
        f"{report.bootstrap_ci.total_return_p05 * 100:+.2f}%" if report.bootstrap_ci else "—"
    )
    table.add_row(
        "Bootstrap lower-5%", boot_lower, "> 0", "✓" if report.gate_bootstrap_lower else "✗"
    )
    n_tested = sum(1 for r in report.regime_breakdown if r.n_days >= 30)
    table.add_row(
        "Regime stress (positive/tested)",
        f"{report.n_positive_regimes}/{n_tested}" + (" (of 5 defined)" if n_tested < 5 else ""),
        "≥ 50%",
        "✓" if report.gate_regime else "✗",
    )
    if report.holdout is not None:
        table.add_row(
            f"Holdout {report.holdout.start}→{report.holdout.end} total return",
            f"{report.holdout.total_return:+.2%}",
            "> 0",
            "✓" if report.gate_holdout else "✗",
        )
    console.print(table)

    if report.cost_sensitivity:
        cost_table = Table(title="Cost-sensitivity sweep (OOS)", show_header=True)
        cost_table.add_column("Slippage bps", justify="right")
        cost_table.add_column("Total return", justify="right")
        cost_table.add_column("Sharpe", justify="right")
        cost_table.add_column("Max DD", justify="right")
        for row in report.cost_sensitivity:
            cost_table.add_row(
                f"{row.slippage_bps:g}",
                f"{row.total_return:+.2%}",
                f"{row.sharpe:.2f}",
                f"{row.max_drawdown:.2%}",
            )
        console.print(cost_table)

    console.print(f"\n[bold]Overall: {'PASS' if report.passed else 'FAIL'}[/]")
    console.print(f"Tear-sheet: {html_path}")
    console.print(f"Validation JSON: {validation_json}")

    if not report.passed:
        raise SystemExit(2)


@cli.command(help="Run today's live rebalance across all enabled strategies.")
@click.option("--dry-run", is_flag=True, help="Print orders only; do not submit.")
@click.option("--asof", default=None, help="Override the rebalance date (YYYY-MM-DD).")
@click.option(
    "--strategy",
    "strategy_filter",
    default=None,
    help="Only rebalance the named strategy (instead of all live-enabled).",
)
@click.option(
    "--include-quarantined",
    is_flag=True,
    help="Dry-run only: include quarantined strategies for observation.",
)
@click.option(
    "--winddown-participation",
    default=0.10,
    show_default=True,
    type=float,
    help="Max fraction of trailing dollar-ADV per orphan wind-down exit order.",
)
@click.option(
    "--derisk-actuate/--no-derisk-actuate",
    default=False,
    show_default=True,
    help="Apply the engine-driven one-way de-risk overlay to gross exposure "
    "(default: shadow — computed and reported, not applied).",
)
@click.option(
    "--leverage",
    type=float,
    default=None,
    help="Deploy the normalized allocation at this total gross (e.g. 1.5). "
    "Hard-capped at 2.0x; de-risk overlay still scales down; Guard-5 fails closed "
    "if breached. Omit to keep today's behaviour (allocation deployed as-is).",
)
def rebalance(
    dry_run: bool,
    asof: str | None,
    strategy_filter: str | None,
    include_quarantined: bool,
    winddown_participation: float,
    derisk_actuate: bool,
    leverage: float | None,
) -> None:
    from quant.live import run_rebalance
    from quant.live.derisk import DeriskConfig

    if include_quarantined and not dry_run:
        raise click.ClickException("--include-quarantined is allowed only with --dry-run.")

    asof_date = date.fromisoformat(asof) if asof else date.today()
    strategies_arg = [strategy_filter] if strategy_filter else None
    report = run_rebalance(
        asof=asof_date,
        dry_run=dry_run,
        strategies=strategies_arg,
        include_quarantined=include_quarantined,
        winddown_participation=winddown_participation,
        derisk_config=DeriskConfig(actuate=derisk_actuate),
        target_leverage=leverage,
    )

    header = Table(title=f"Rebalance {report.asof} — {'DRY RUN' if dry_run else 'LIVE'}")
    header.add_column("Field")
    header.add_column("Value", justify="right")
    header.add_row("Account equity", f"${report.equity:,.2f}")
    header.add_row("Enabled strategies", str(len(report.enabled_strategies)))
    header.add_row("Total orders", str(report.total_orders))
    if report.leverage is not None:
        header.add_row("Portfolio leverage", f"x{report.leverage:.2f} gross target")
    console.print(header)
    if report.derisk is not None:
        d = report.derisk
        mult = d.get("multiplier", 1.0)
        if d.get("degraded"):
            why = ", ".join(d.get("reasons") or ["no engine state"])
            console.print(f"[dim]de-risk overlay: standby — {why}[/dim]")
        elif d.get("reasons"):
            tag = (
                "[red]APPLIED[/red]"
                if d.get("actuated")
                else "[yellow]SHADOW (not applied)[/yellow]"
            )
            console.print(f"de-risk overlay: gross x{mult:.2f} {tag} <- {', '.join(d['reasons'])}")
        else:
            console.print(
                f"[green]de-risk overlay: neutral (gross x{mult:.2f}, engine risk-on)[/green]"
            )
    if report.skipped_reason:
        console.print(f"[yellow]Skipped: {report.skipped_reason}[/yellow]")

    if not report.outcomes:
        console.print("[dim]No outcomes.[/dim]")
        return

    detail = Table(title="Per-strategy outcomes", show_header=True)
    for col in ("Strategy", "Target", "Previous", "Orders", "Error"):
        detail.add_column(col)
    for outcome in report.outcomes:
        detail.add_row(
            outcome.slug,
            str(len(outcome.target)),
            str(len(outcome.previous)),
            str(len(outcome.orders)),
            outcome.error or "",
        )
    console.print(detail)

    if report.winddown_outcomes:
        wd = Table(title="Orphan wind-down (exit-only)", show_header=True)
        wd.add_column("Strategy")
        wd.add_column("Exited", justify="right")
        wd.add_column("Remaining", justify="right")
        wd.add_column("Note")
        for o in report.winddown_outcomes:
            exited = ", ".join(f"{s}:{q}" for s, q in sorted(o.exited.items())) or "—"
            remaining = ", ".join(f"{s}:{q}" for s, q in sorted(o.remaining.items()) if q) or "flat"
            note = o.error or (("skipped: " + ",".join(o.skipped)) if o.skipped else "")
            wd.add_row(o.slug, exited, remaining, note)
        console.print(wd)


@cli.command(help="Open the HTML tear-sheet for <strategy> in your default browser.")
@click.argument("strategy")
def tearsheet(strategy: str) -> None:
    settings = Settings()  # type: ignore[call-arg]
    path = settings.data_dir / "backtests" / strategy / "tearsheet.html"
    if not path.exists():
        raise click.ClickException(
            f"No tearsheet at {path}. Run `quant backtest {strategy}` first."
        )
    webbrowser.open(path.resolve().as_uri())
    console.print(f"Opened {path}")


@cli.command(help="Pre-flight check before connecting Alpaca for paper trading.")
def doctor() -> None:
    """Run a series of environment + state checks. Exit 0 = ready; >0 = fix needed."""
    from quant.live.safety import (
        check_bar_freshness,
        check_market_open,
        check_reconciliation,
        check_risk_limits,
    )

    checks: list[tuple[str, bool, str]] = []

    # 1. Settings load (.env present + required keys).
    try:
        s = Settings()  # type: ignore[call-arg]
        cfg_ok = bool(s.alpaca_api_key) and bool(s.alpaca_secret_key) and bool(s.fred_api_key)
        checks.append(
            (
                "config",
                cfg_ok,
                f"data_dir={s.data_dir}; alpaca_paper={s.alpaca_paper}",
            )
        )
        settings_obj: Settings | None = s if cfg_ok else None
    except Exception as exc:
        checks.append(("config", False, f"Settings() raised: {exc!r}"))
        settings_obj = None

    governance_live_slugs: list[str] = []
    if settings_obj is not None:
        governance_live_slugs, governance_error = _doctor_governance_live_slugs(
            settings_obj.data_dir
        )
        checks.append(
            (
                "governance",
                governance_error is None,
                (
                    f"{len(governance_live_slugs)} live: {', '.join(governance_live_slugs)}"
                    if governance_error is None
                    else governance_error
                ),
            )
        )

    # 2. Alpaca connectivity (account fetch).
    alpaca_positions: list[object] = []
    if settings_obj is not None:
        try:
            client = AlpacaClient(settings=settings_obj)
            acct = client.account()
            alpaca_positions = list(client.positions())
            checks.append(
                (
                    "alpaca_connectivity",
                    True,
                    f"equity=${acct.equity:,.2f}; paper={settings_obj.alpaca_paper}",
                )
            )
        except Exception as exc:
            checks.append(("alpaca_connectivity", False, f"{exc!r}"))

    # 3. Trading-day check.
    market = check_market_open(date.today())
    checks.append(("market_open", market.ok, market.detail))

    # 4. Bar cache freshness.
    if settings_obj is not None:
        # Sample one strategy's universe — pick the smallest one.
        sample_slug = governance_live_slugs[:1]
        sample_symbols = list(REGISTRY[sample_slug[0]].spec.universe) if sample_slug else []
        if sample_symbols:
            fresh = check_bar_freshness(
                settings_obj.data_dir, symbols=sample_symbols, asof=date.today()
            )
            checks.append(("bar_freshness", fresh.ok, fresh.detail))

    # 5. Reconciliation guard (informational on first run).
    if settings_obj is not None:
        recon = check_reconciliation(
            data_dir=settings_obj.data_dir,
            alpaca_positions=alpaca_positions,  # type: ignore[arg-type]
            enabled_slugs=governance_live_slugs,
        )
        checks.append(("reconciliation", recon.ok, recon.detail))

    # 6. Risk limits.
    if settings_obj is not None:
        risk = check_risk_limits(
            data_dir=settings_obj.data_dir, enabled_slugs=governance_live_slugs
        )
        checks.append(("risk_limits", risk.ok, risk.detail))

    # Render table.
    table = Table(title="quant doctor", show_header=True)
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    n_pass = 0
    for name, ok, detail in checks:
        status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        table.add_row(name, status, detail)
        n_pass += int(ok)
    console.print(table)
    console.print(
        f"\n[bold]{n_pass}/{len(checks)} checks passed[/]"
        + (
            " — ready to connect Alpaca for paper trading."
            if n_pass == len(checks)
            else " — fix the failing checks above before going live."
        )
    )
    if n_pass < len(checks):
        raise SystemExit(1)


@cli.command(help="Print the structured trade journal.")
@click.option("--since", default=None, help="Filter trades since YYYY-MM-DD.")
@click.option("--strategy", default=None, help="Filter trades by strategy slug.")
@click.option("--limit", default=50, show_default=True, type=int, help="Cap rows printed.")
def journal(since: str | None, strategy: str | None, limit: int) -> None:
    from quant.live import read_journal

    settings = Settings()  # type: ignore[call-arg]
    since_date = date.fromisoformat(since) if since else None
    df = read_journal(settings.data_dir, since=since_date, strategy=strategy)
    if df.empty:
        console.print("[dim]No trades found.[/dim]")
        return

    df = df.tail(limit)
    table = Table(title=f"Trade journal ({len(df)} rows)", show_header=True)
    for col in ("date", "strategy", "symbol", "side", "qty", "client_order_id", "dry_run"):
        table.add_column(col)
    for row in df.itertuples(index=False):
        ts = pd.Timestamp(str(row.date))
        table.add_row(
            ts.date().isoformat(),
            str(row.strategy),
            str(row.symbol),
            str(row.side),
            str(row.qty),
            str(row.client_order_id),
            str(bool(row.dry_run)),
        )
    console.print(table)


@cli.command(help="Open the Textual TUI monitor.")
def monitor() -> None:
    from quant.tui import QuantMonitor

    QuantMonitor().run()


def _governance_state_labels(data_dir: Path) -> dict[str, str]:
    from quant.governance.models import GovernanceError
    from quant.governance.store import load_strategy_states, strategy_states_path

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError:
        return {}
    return {slug: state.state.value for slug, state in states.items()}


@cli.group(help="Strategy governance and evidence-gated live eligibility.")
def governance() -> None:
    pass


@governance.command("refresh", help="Recompute governance artifacts from validation evidence.")
@click.option("--asof", default=None, help="Evaluation date (YYYY-MM-DD). Default: today.")
@click.option("--max-age-days", default=30, show_default=True, type=int)
def governance_refresh(asof: str | None, max_age_days: int) -> None:
    from quant.governance import GovernancePolicy, build_governance_artifacts

    settings = Settings()  # type: ignore[call-arg]
    asof_date = date.fromisoformat(asof) if asof else date.today()
    states = build_governance_artifacts(
        data_dir=settings.data_dir,
        registry=REGISTRY,
        policy=GovernancePolicy(max_validation_age_days=max_age_days),
        asof=asof_date,
    )
    table = Table(title=f"Governance refresh — {asof_date}", show_header=True)
    table.add_column("Strategy")
    table.add_column("State")
    table.add_column("Reason")
    for slug, state in sorted(states.items()):
        table.add_row(slug, state.state.value, state.reason)
    console.print(table)


@governance.command("status", help="Show current governance state for each strategy.")
def governance_status() -> None:
    from quant.governance.allocation import allocate_capital
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import (
        drift_report_path,
        load_strategy_states,
        load_validation_manifest,
        strategy_states_path,
        validation_manifest_path,
    )

    settings = Settings()  # type: ignore[call-arg]
    try:
        states = load_strategy_states(strategy_states_path(settings.data_dir))
    except GovernanceError as exc:
        states = {}
        console.print(f"[yellow]{exc}; run `quant governance refresh`.[/yellow]")
    try:
        evidence = load_validation_manifest(validation_manifest_path(settings.data_dir))
    except GovernanceError:
        evidence = {}
    allocation = allocate_capital(states, evidence_by_slug=evidence)
    drift_flags: dict[str, str] = {}
    drift_path = drift_report_path(settings.data_dir)
    if drift_path.exists():
        try:
            payload = json.loads(drift_path.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            if isinstance(rows, list):
                priority = {"halt_candidate": 3, "watch": 2, "normal": 1}
                for raw in rows:
                    if not isinstance(raw, dict):
                        continue
                    slug = raw.get("strategy")
                    flag = raw.get("flag")
                    if not isinstance(slug, str) or not isinstance(flag, str):
                        continue
                    current = drift_flags.get(slug, "normal")
                    if priority.get(flag, 0) >= priority.get(current, 0):
                        drift_flags[slug] = flag
        except Exception:
            drift_flags = {}

    table = Table(title="Strategy governance", show_header=True)
    for col in (
        "Strategy",
        "Code Live",
        "Governance",
        "Age",
        "Allocation",
        "Drift",
        "Why no trade",
    ):
        table.add_column(col)
    for spec in list_strategies():
        state = states.get(spec.slug)
        if state is None:
            table.add_row(
                spec.slug,
                "yes" if spec.enabled_live else "no",
                GovernanceState.UNKNOWN.value,
                "",
                "0.0%",
                drift_flags.get(spec.slug, "unknown"),
                "no governance artifact",
            )
            continue
        age = "" if state.validation_age_days is None else f"{state.validation_age_days}d"
        # A shielded LIVE incumbent is loud: it is trading on retained authority
        # across a methodology bump and needs human review, not "eligible".
        if state.shielded:
            governance_cell = f"{state.state.value} ⚠SHIELDED ({state.shield_consecutive})"
            why_no_trade = state.reason
        elif state.state is GovernanceState.LIVE:
            governance_cell = state.state.value
            why_no_trade = "eligible"
        else:
            governance_cell = state.state.value
            why_no_trade = state.reason
        table.add_row(
            spec.slug,
            "yes" if spec.enabled_live else "no",
            governance_cell,
            age,
            f"{allocation.get(spec.slug, 0.0) * 100:.1f}%",
            drift_flags.get(spec.slug, "unknown"),
            why_no_trade,
        )
    console.print(table)


@governance.command("audit", help="Show reproducibility metadata for a strategy.")
@click.argument("strategy")
def governance_audit(strategy: str) -> None:
    from quant.governance.audit import build_validation_audit

    _require_strategy(strategy)
    settings = Settings()  # type: ignore[call-arg]
    audit = build_validation_audit(settings.data_dir, strategy, repo_dir=Path.cwd())

    table = Table(title=f"Validation audit — {strategy}", show_header=True)
    table.add_column("Field")
    table.add_column("Value")

    def value(raw: object) -> str:
        if raw is None or raw == "":
            return "[yellow]missing[/yellow]"
        return str(raw)

    table.add_row("Git SHA", value(audit.git_sha))
    table.add_row("Validation command", value(audit.validation_command))
    table.add_row("Data range", f"{value(audit.data_range[0])} .. {value(audit.data_range[1])}")
    table.add_row("Bootstrap resamples", value(audit.bootstrap_resamples))
    table.add_row("Bootstrap seed", value(audit.bootstrap_seed))
    table.add_row("Validation report SHA-256", value(audit.validation_report_hash))
    table.add_row("Chosen params SHA-256", value(audit.chosen_params_hash))
    table.add_row("Walk-forward parquet SHA-256", value(audit.walkforward_parquet_hash))
    table.add_row("Governance state", value(audit.governance_state))
    table.add_row("Reason codes", ", ".join(audit.reason_codes) or "ok")
    table.add_row("Missing artifacts", ", ".join(audit.missing_artifacts) or "none")
    console.print(table)
    console.print(audit.explanation)


@governance.command("drift", help="Show advisory paper-P&L drift flags.")
def governance_drift() -> None:
    from quant.governance.drift import summarize_drift
    from quant.governance.store import drift_report_path
    from quant.live.bookkeeping import read_equity

    settings = Settings()  # type: ignore[call-arg]
    equity = read_equity(settings.data_dir)
    if equity.empty or "equity" not in equity.columns:
        console.print("[yellow]No paper equity history available yet.[/yellow]")
        return
    returns = equity["equity"].astype(float).pct_change(fill_method=None).dropna()
    if returns.empty:
        console.print("[yellow]Not enough paper equity history for drift analysis.[/yellow]")
        return
    realized = {"account": returns}
    expected = {"account": pd.Series(0.0, index=returns.index)}
    rows = summarize_drift(realized, expected)
    path = drift_report_path(settings.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_date": date.today().isoformat(),
                "rows": [
                    {
                        "strategy": row.strategy,
                        "window": row.window,
                        "realized_return": row.realized_return,
                        "expected_return": row.expected_return,
                        "z_score": row.z_score,
                        "flag": row.flag,
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table = Table(title="Paper P&L drift", show_header=True)
    for col in ("Strategy", "Window", "Realized", "Expected", "Z", "Flag"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            row.strategy,
            str(row.window),
            f"{row.realized_return:+.2%}",
            f"{row.expected_return:+.2%}",
            f"{row.z_score:+.2f}",
            row.flag,
        )
    console.print(
        table if rows else "[yellow]Not enough paper equity history for drift windows.[/yellow]"
    )
    console.print(f"[dim]wrote {path}[/dim]")


@governance.command("halt", help="Emergency stop: block all non-dry-run paper orders.")
@click.option("--reason", required=True, help="Operator reason recorded in governance/halt.json.")
def governance_halt(reason: str) -> None:
    from quant.governance.halt import set_halt

    settings = Settings()  # type: ignore[call-arg]
    state = set_halt(settings.data_dir, reason=reason)
    console.print(f"[red]Governance halted[/red] at {state.updated_at.isoformat()}: {state.reason}")


@governance.command("resume", help="Resume paper orders after an emergency halt.")
@click.option("--reason", required=True, help="Operator reason recorded in governance/halt.json.")
def governance_resume(reason: str) -> None:
    from quant.governance.halt import clear_halt

    settings = Settings()  # type: ignore[call-arg]
    state = clear_halt(settings.data_dir, reason=reason)
    console.print(
        f"[green]Governance resumed[/green] at {state.updated_at.isoformat()}: {state.reason}"
    )


@cli.group(help="Data subcommands.")
def data() -> None:
    pass


@data.command("refresh", help="Refresh bar caches for ETFs + S&P 500 + registered universes.")
@click.option("--start", default="2010-01-01", show_default=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD). Default: today.")
def data_refresh(start: str, end: str | None) -> None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    console.print(f"[bold]Refreshing caches over {start_date}..{end_date}...[/bold]")
    report = refresh_caches(start=start_date, end=end_date)
    table = Table(title="Refresh report")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("symbols_fetched", str(report.symbols_fetched))
    table.add_row("rows_total", str(report.rows_total))
    table.add_row("elapsed_s", f"{report.elapsed_s:.1f}")
    table.add_row("errors", str(len(report.errors)))
    console.print(table)
    if report.errors:
        console.print("[red]First 5 errors:[/red]")
        for err in report.errors[:5]:
            console.print(f"  {err}")


@data.command(
    "refresh-fundamentals",
    help="Pull SEC EDGAR fundamentals for the multi-factor strategy universe.",
)
def data_refresh_fundamentals() -> None:
    """Download EDGAR /companyfacts for every name in the multi-factor universe."""
    from quant.data.edgar import fetch_company_facts

    if "multi-factor" not in REGISTRY:
        raise click.ClickException("multi-factor strategy is not registered.")
    universe = list(REGISTRY["multi-factor"].spec.universe)
    table = Table(title=f"EDGAR refresh — {len(universe)} symbols", show_header=True)
    table.add_column("Symbol")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    n_ok = 0
    for sym in universe:
        try:
            df = fetch_company_facts(sym)
            status = "[green]OK[/]" if not df.empty else "[yellow]empty[/]"
            table.add_row(sym, status, str(len(df)))
            if not df.empty:
                n_ok += 1
        except Exception as exc:  # network flake / SEC throttle
            table.add_row(sym, "[red]FAIL[/]", f"{exc!r}")
    console.print(table)
    console.print(f"[bold]{n_ok}/{len(universe)} symbols cached.[/]")


@data.command("inventory", help="Show what's currently on disk under data/.")
def data_inventory() -> None:
    settings = Settings()  # type: ignore[call-arg]
    base = Path(settings.data_dir)
    table = Table(title=f"Data inventory ({base})", show_header=True)
    table.add_column("Subdirectory")
    table.add_column("Files", justify="right")
    table.add_column("Size (MB)", justify="right")
    for sub in ("universe", "raw", "backtests", "live", "features", "fundamentals", "macro"):
        d = base / sub
        if not d.exists():
            table.add_row(sub, "0", "0.00")
            continue
        files = [f for f in d.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        table.add_row(sub, str(len(files)), f"{size_mb:.2f}")
    console.print(table)


@data.command("snapshot", help="Create an immutable manifest for raw bar cache inputs.")
@click.option("--symbols", required=True, help="Comma-separated symbols to include.")
@click.option("--start", required=True, help="Requested start date (YYYY-MM-DD).")
@click.option("--end", required=True, help="Requested end date (YYYY-MM-DD).")
@click.option("--snapshot-id", default=None, help="Optional deterministic snapshot id.")
def data_snapshot(symbols: str, start: str, end: str, snapshot_id: str | None) -> None:
    from quant.data.snapshot import create_data_snapshot

    settings = Settings()  # type: ignore[call-arg]
    symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    manifest = create_data_snapshot(
        settings.data_dir,
        symbols=symbol_list,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        snapshot_id=snapshot_id,
    )
    table = Table(title=f"Data snapshot — {manifest.snapshot_id}", show_header=True)
    table.add_column("Symbol")
    table.add_column("Rows", justify="right")
    table.add_column("SHA-256")
    for symbol, row in manifest.symbols.items():
        table.add_row(symbol, str(row.rows), row.sha256[:12] if row.sha256 else "missing")
    console.print(table)


@data.command("quality", help="Run daily bar cache quality checks and write ops health.")
@click.option("--symbols", default=None, help="Comma-separated symbols. Default: all raw caches.")
@click.option("--start", default="2010-01-01", show_default=True)
@click.option(
    "--end", default=None, help="End date (YYYY-MM-DD). Default: last completed trading day."
)
def data_quality(symbols: str | None, start: str, end: str | None) -> None:
    from quant.data.quality import evaluate_bar_quality

    settings = Settings()  # type: ignore[call-arg]
    raw_dir = settings.data_dir / "raw"
    if symbols:
        symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    else:
        symbol_list = sorted(path.stem.upper() for path in raw_dir.glob("*.parquet"))
    frames = {}
    for symbol in symbol_list:
        path = raw_dir / f"{symbol}.parquet"
        if path.exists():
            frames[symbol] = pd.read_parquet(path)
        else:
            frames[symbol] = pd.DataFrame()
    end_date = date.fromisoformat(end) if end else _default_data_quality_end_date(date.today())
    report = evaluate_bar_quality(
        frames,
        start=date.fromisoformat(start),
        end=end_date,
    )
    out = settings.data_dir / "ops" / "health" / "data_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": report.passed,
                "start": report.start.isoformat(),
                "end": report.end.isoformat(),
                "symbols": {
                    symbol: {
                        "rows": row.rows,
                        "missing_bars": row.missing_bars,
                        "duplicate_timestamps": row.duplicate_timestamps,
                        "impossible_ohlc": row.impossible_ohlc,
                        "stale": row.stale,
                        "passed": row.passed,
                    }
                    for symbol, row in report.symbols.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table = Table(title="Data quality", show_header=True)
    for col in ("Symbol", "Rows", "Missing", "Dupes", "Bad OHLC", "Stale", "Pass"):
        table.add_column(col)
    for symbol, row in report.symbols.items():
        table.add_row(
            symbol,
            str(row.rows),
            str(row.missing_bars),
            str(row.duplicate_timestamps),
            str(row.impossible_ohlc),
            "yes" if row.stale else "no",
            "yes" if row.passed else "no",
        )
    console.print(table)
    console.print(f"[dim]wrote {out}[/dim]")
    if not report.passed:
        raise SystemExit(2)


@cli.group(help="Research experiment registry and comparison.")
def research() -> None:
    pass


def _experiments_path() -> Path:
    settings = Settings()  # type: ignore[call-arg]
    return settings.data_dir / "research" / "experiments.jsonl"


@research.command("list", help="List recorded research/backtest/validation experiments.")
def research_list() -> None:
    from quant.research.registry import list_experiments

    rows = list_experiments(_experiments_path())
    table = Table(title="Research experiments", show_header=True)
    for col in ("Run ID", "Strategy", "Kind", "Created", "DSR", "Overall"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            row.run_id,
            row.strategy,
            row.kind,
            row.created_at.isoformat(),
            f"{row.metrics.get('dsr', float('nan')):.4f}" if "dsr" in row.metrics else "—",
            str(row.gates.get("overall", "")),
        )
    console.print(table)


@research.command("show", help="Show one experiment as JSON.")
@click.argument("run_id")
def research_show(run_id: str) -> None:
    from quant.research.registry import list_experiments

    for row in list_experiments(_experiments_path()):
        if row.run_id == run_id:
            console.print_json(json.dumps(row.to_json_dict(), sort_keys=True))
            return
    raise click.ClickException(f"Unknown experiment {run_id!r}")


@research.command("compare", help="Compare metrics between two experiments.")
@click.argument("left")
@click.argument("right")
def research_compare(left: str, right: str) -> None:
    from quant.research.registry import compare_experiments

    comparison = compare_experiments(_experiments_path(), left, right)
    table = Table(title=f"Experiment comparison {left} -> {right}", show_header=True)
    table.add_column("Metric")
    table.add_column("Delta", justify="right")
    metric_delta = comparison["metric_delta"]
    if isinstance(metric_delta, dict):
        for metric, delta in sorted(metric_delta.items()):
            table.add_row(str(metric), f"{float(delta):+0.4f}")
    console.print(table)


@research.command("leaderboard", help="Rank experiments by a metric.")
@click.option("--metric", default="dsr", show_default=True)
def research_leaderboard(metric: str) -> None:
    from quant.research.registry import leaderboard

    rows = leaderboard(_experiments_path(), metric=metric)
    table = Table(title=f"Research leaderboard — {metric}", show_header=True)
    for col in ("Run ID", "Strategy", "Metric"):
        table.add_column(col)
    for row in rows:
        table.add_row(row.run_id, row.strategy, f"{row.metrics[metric]:.4f}")
    console.print(table)


@research.command(
    "signals",
    help="Compute today's trailing-only quant signal battery and append it to "
    "data/research/signals.jsonl. Read-only/advisory; always exits 0.",
)
@click.option("--asof", default=None, help="ISO date; default today.")
@click.option("--dry-run", is_flag=True, default=False, help="Compute + print, do not log.")
def research_signals(asof: str | None, dry_run: bool) -> None:
    # Whole body guarded: this runs as an unattended job, so it MUST NOT raise
    # (a non-zero exit pages off-box and suppresses the tick heartbeat).
    try:
        from quant.research.signals import (
            append_signals,
            load_market_signals,
            render_signals,
            signals_path,
            to_json_dict,
        )
        from quant.util.atomic import write_json_atomic

        settings = Settings()  # type: ignore[call-arg]
        d = date.fromisoformat(asof) if asof else date.today()
        rec = load_market_signals(settings.data_dir, d)
        console.print(render_signals(rec))
        if not dry_run and rec.computable:
            append_signals(signals_path(settings.data_dir), rec)
            write_json_atomic(
                settings.data_dir / "research" / "signals_latest.json", to_json_dict(rec)
            )
            console.print(f"[dim]appended {signals_path(settings.data_dir)}[/dim]")
    except Exception as exc:  # advisory job: degrade, never page
        console.print("Research signals: unavailable")
        logger.info("research signals CLI failed ({!r})", exc)


@research.command("signals-show", help="Print the latest logged signal record as JSON.")
def research_signals_show() -> None:
    from quant.research.signals import read_latest_signals, signals_path, to_json_dict

    settings = Settings()  # type: ignore[call-arg]
    rec = read_latest_signals(signals_path(settings.data_dir))
    if rec is None:
        raise click.ClickException("No signals logged yet.")
    console.print_json(json.dumps(to_json_dict(rec), sort_keys=True))


@cli.group(help="Continuous market-state engine — always-on, READ-ONLY; never trades or halts.")
def engine() -> None:
    pass


@engine.command(
    "run",
    help="Run the continuous engine loop: maintain live MarketState + emit events. "
    "Read-only/advisory — places no orders, sets no halt. --once/--max-cycles bound it.",
)
@click.option("--once", is_flag=True, default=False, help="Run a single cycle and exit.")
@click.option(
    "--max-cycles", type=int, default=None, help="Stop after N cycles (default: forever)."
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Compute + log, post nothing, no Claude."
)
def engine_run(once: bool, max_cycles: int | None, dry_run: bool) -> None:
    from quant.engine import run_engine

    settings = Settings()  # type: ignore[call-arg]
    run_engine(
        settings,
        once=once,
        max_cycles=max_cycles,
        dry_run=dry_run,
        console_print=lambda s: console.print(f"[dim]{s}[/dim]"),
    )


@engine.command("state", help="Print the latest persisted MarketState snapshot.")
def engine_state() -> None:
    from quant.engine.loop import engine_dir

    settings = Settings()  # type: ignore[call-arg]
    path = engine_dir(settings.data_dir) / "state.json"
    if not path.exists():
        raise click.ClickException("No engine state yet — run `quant engine run --once`.")
    console.print_json(path.read_text(encoding="utf-8"))


@engine.command("status", help="Print the engine heartbeat + the most recent events.")
def engine_status() -> None:
    from quant.engine.loop import engine_dir

    settings = Settings()  # type: ignore[call-arg]
    edir = engine_dir(settings.data_dir)
    hb = edir / "heartbeat.json"
    if hb.exists():
        console.print("[bold]heartbeat[/bold]")
        console.print_json(hb.read_text(encoding="utf-8"))
    else:
        console.print("[yellow]no heartbeat yet[/yellow]")
    events = edir / "events.jsonl"
    if events.exists():
        lines = [ln for ln in events.read_text(encoding="utf-8").splitlines() if ln.strip()]
        console.print(f"[bold]recent events[/bold] ({len(lines)} total)")
        for ln in lines[-10:]:
            console.print(f"  {ln}")


@cli.group(help="Market news + local sentiment (read-only; no LLM in scoring).")
def news() -> None:
    pass


@news.command("sentiment", help="Fetch recent headlines and print the local sentiment read.")
@click.option("--hours", type=int, default=4, show_default=True, help="Lookback window.")
@click.option("--limit", type=int, default=50, show_default=True, help="Max headlines.")
def news_sentiment(hours: int, limit: int) -> None:
    from quant.data.news import fetch_news
    from quant.nlp.sentiment import render_sentiment, score_news, score_text

    settings = Settings()  # type: ignore[call-arg]
    items = fetch_news(settings, lookback_minutes=hours * 60, limit=limit)
    s = score_news(items)
    console.print(render_sentiment(s))
    for it in sorted(items, key=lambda i: score_text(i.headline))[:8]:
        console.print(f"  [{score_text(it.headline):+.2f}] {it.headline[:100]}")


@cli.group(help="Macro / policy / event-risk (scheduled events + FRED uncertainty).")
def macro() -> None:
    pass


@macro.command(
    "eventrisk", help="Print the current macro/policy/event-risk read + upcoming events."
)
def macro_eventrisk() -> None:
    from quant.macro.events import live_event_risk, render_event_risk, upcoming_events

    settings = Settings()  # type: ignore[call-arg]
    today = date.today()
    console.print(render_event_risk(live_event_risk(settings, today)))
    console.print("[bold]upcoming events[/bold]")
    for ev in upcoming_events(today, horizon_days=30):
        console.print(f"  {ev.date}  [{ev.impact}]  {ev.name}")


@macro.command("nowcast", help="Print the macro / business-cycle nowcast (FRED curve/credit/Sahm).")
def macro_nowcast() -> None:
    from quant.macro.nowcast import live_macro_nowcast, render_macro_nowcast

    settings = Settings()  # type: ignore[call-arg]
    n = live_macro_nowcast(settings, date.today())
    console.print(render_macro_nowcast(n))
    console.print(
        f"  recession_signal={n.recession_signal}  components={n.n_components}  "
        f"breakeven10={n.breakeven_10y}  claims={n.initial_claims}"
    )


@cli.group(help="Fundamentals: cross-sectional value/quality on the mega-cap universe (SEC EDGAR).")
def fundamentals() -> None:
    pass


@fundamentals.command(
    "status", help="Print the live PIT fundamentals read + per-name value/quality factors."
)
@click.option("--asof", default=None, help="Query date (YYYY-MM-DD). Default: today.")
@click.option(
    "--symbols", default=None, help="Comma-separated tickers. Default: mega-cap universe."
)
def fundamentals_status(asof: str | None, symbols: str | None) -> None:
    from quant.fundamentals.factors import (
        FundamentalsConfig,
        compute_fundamentals,
        fundamental_rows,
        render_fundamentals,
    )

    settings = Settings()  # type: ignore[call-arg]
    asof_date = date.fromisoformat(asof) if asof else date.today()
    cfg = (
        FundamentalsConfig(
            universe=tuple(s.strip().upper() for s in symbols.split(",") if s.strip())
        )
        if symbols
        else FundamentalsConfig()
    )
    rows = fundamental_rows(settings, asof_date, config=cfg)
    console.print(render_fundamentals(compute_fundamentals(rows, asof=asof_date, config=cfg)))
    console.print("[bold]per-name factors[/bold] (E/P, B/M, gross-profit, asset-growth)")
    for r in rows:
        ey = f"{r.earnings_yield:+.1%}" if r.earnings_yield is not None else "  n/a"
        btm = f"{r.book_to_market:.2f}" if r.book_to_market is not None else " n/a"
        gp = f"{r.gross_profitability:.2f}" if r.gross_profitability is not None else " n/a"
        ag = f"{r.asset_growth:+.1%}" if r.asset_growth is not None else "  n/a"
        console.print(f"  {r.symbol:<6} EY={ey}  BM={btm}  GP={gp}  AG={ag}")


@cli.group(help="Forecasting models (Phase 8, research/shadow) — HAR-RV vol, evaluated OOS.")
def forecast() -> None:
    pass


@forecast.command("vol", help="Live one-day-ahead HAR-RV volatility forecast (annualized).")
@click.option("--symbol", default="SPY", show_default=True, help="Underlying symbol.")
def forecast_vol(symbol: str) -> None:
    from quant.forecast.vol import OOS_SKILL_SPY, live_vol_forecast, render_vol_forecast

    settings = Settings()  # type: ignore[call-arg]
    skill = OOS_SKILL_SPY if symbol.upper() == "SPY" else None
    f = live_vol_forecast(settings, date.today(), symbol=symbol.upper(), oos_skill=skill)
    console.print(render_vol_forecast(f))


@forecast.command(
    "vol-eval",
    help="Walk-forward OOS eval: HAR/GARCH/GJR vs EWMA/RW/rolling (QLIKE + MSE + Diebold-Mariano).",
)
@click.option("--symbol", default="SPY", show_default=True)
@click.option("--min-train", default=504, show_default=True, type=int, help="Initial train days.")
@click.option(
    "--no-garch", is_flag=True, default=False, help="Skip the GARCH-family competitors (faster)."
)
def forecast_vol_eval(symbol: str, min_train: int, no_garch: bool) -> None:
    import pandas as pd

    from quant.data import bars
    from quant.forecast.vol import walk_forward_eval

    settings = Settings()  # type: ignore[call-arg]
    path = bars._cache_path(symbol.upper(), settings.data_dir)
    if not path.exists():
        raise click.ClickException(
            f"No cached bars for {symbol.upper()} — run `quant data refresh`."
        )
    close = pd.read_parquet(path)["close"].dropna().to_numpy()
    ev = walk_forward_eval(close, min_train=min_train, include_garch=not no_garch)
    console.print(
        f"[bold]{symbol.upper()} vol-forecast OOS[/bold] — {ev.n_oos} days, winner: {ev.winner}"
    )
    for _m, s in sorted(ev.scores.items(), key=lambda kv: kv[1].mean_qlike):
        console.print(
            f"  {s.model:<8} QLIKE mean={s.mean_qlike:.4f} med={s.median_qlike:.4f}  "
            f"MSE={s.mean_mse:.2e}  n={s.n}"
        )
    if ev.dm_stat is not None and ev.dm_pvalue is not None:
        sig = ev.dm_pvalue < 0.05
        verdict = (
            "HAR beats EWMA"
            if (ev.dm_stat < 0 and sig)
            else "EWMA beats HAR"
            if (ev.dm_stat > 0 and sig)
            else "no significant difference"
        )
        console.print(f"  DM(HAR vs EWMA): stat={ev.dm_stat:+.3f} p={ev.dm_pvalue:.4f} → {verdict}")
    if ev.dm_garch_har_stat is not None and ev.dm_garch_har_pvalue is not None:
        sig = ev.dm_garch_har_pvalue < 0.05
        verdict = (
            "GARCH beats HAR"
            if (ev.dm_garch_har_stat < 0 and sig)
            else "HAR beats GARCH"
            if (ev.dm_garch_har_stat > 0 and sig)
            else "no significant difference"
        )
        console.print(
            f"  DM(GARCH vs HAR): stat={ev.dm_garch_har_stat:+.3f} "
            f"p={ev.dm_garch_har_pvalue:.4f} → {verdict}"
        )
    console.print(
        "[dim]Research-only: vol forecasts are advisory/shadow and drive no sizing "
        "until a model beats the incumbent OOS and is consciously promoted.[/dim]"
    )


def _factor_closes(settings: Settings) -> Any:
    import pandas as pd

    from quant.data import bars
    from quant.forecast.factor import FACTOR_UNIVERSE

    frames = {}
    for sym in FACTOR_UNIVERSE:
        path = bars._cache_path(sym, settings.data_dir)
        if path.exists():
            df = pd.read_parquet(path)
            if "close" in df.columns and len(df):
                frames[sym] = df["close"]
    if not frames:
        raise click.ClickException(
            "No cached bars for the factor universe — run `quant data refresh`."
        )
    return pd.DataFrame(frames).sort_index()


@forecast.command(
    "factor", help="Current cross-sectional factor scores + top/bottom names (research)."
)
@click.option("--top", default=8, show_default=True, type=int, help="Names to list each side.")
def forecast_factor(top: int) -> None:
    from quant.forecast.factor import compute_factor_scores, render_factor_scores

    settings = Settings()  # type: ignore[call-arg]
    closes = _factor_closes(settings)
    f = compute_factor_scores(closes, date.today(), data_dir=settings.data_dir, top_n=top)
    console.print(render_factor_scores(f))
    console.print(
        "[dim]RESEARCH ONLY — the equal-weight composite had NEGATIVE OOS IC on large-caps "
        "2010-26 (factor winter; only momentum positive). Not a trade signal. See `factor-eval`.[/dim]"
    )


@forecast.command(
    "factor-eval", help="Purged walk-forward cross-sectional IC: composite vs ridge (honest OOS)."
)
@click.option(
    "--model", default="composite", show_default=True, type=click.Choice(["composite", "ridge"])
)
def forecast_factor_eval(model: str) -> None:
    from quant.forecast.factor import walk_forward_factor_eval

    settings = Settings()  # type: ignore[call-arg]
    closes = _factor_closes(settings)
    ev = walk_forward_factor_eval(closes, data_dir=settings.data_dir, model=model)
    console.print(f"[bold]factor model OOS[/bold] ({model}) — {ev.n_periods} monthly periods")
    if ev.mean_ic is not None:
        console.print(f"  mean IC={ev.mean_ic:+.4f} (t={ev.ic_tstat:+.2f}, IR={ev.ic_ir:+.3f})")
    if ev.mean_rank_ic is not None:
        console.print(
            f"  rank IC={ev.mean_rank_ic:+.4f} (t={ev.rank_ic_tstat:+.2f}, hit={ev.hit_rate:.0%})"
        )
    if ev.mean_tertile_spread is not None:
        console.print(f"  top-minus-bottom tertile (21d fwd)={ev.mean_tertile_spread:+.2%}")
    if ev.per_factor_ic:
        ranked = sorted(ev.per_factor_ic.items(), key=lambda kv: -kv[1])
        console.print("  per-factor mean IC: " + ", ".join(f"{k}={v:+.4f}" for k, v in ranked))
    console.print(
        "[dim]Universe is today's large-caps (survivorship-biased) → absolute IC is optimistic; "
        "research-only, not promoted to any tilt.[/dim]"
    )


@forecast.command(
    "gbm-eval",
    help="DSR/PSR-gated gradient-boosting alpha (purged walk-forward, research-only).",
)
def forecast_gbm_eval() -> None:
    from quant.forecast.factor import gbm_research_verdict

    settings = Settings()  # type: ignore[call-arg]
    closes = _factor_closes(settings)
    v = gbm_research_verdict(closes, data_dir=settings.data_dir)
    console.print(f"[bold]GBM alpha OOS[/bold] — {v.n_periods} monthly periods")
    if v.mean_rank_ic is not None:
        console.print(f"  rank IC={v.mean_rank_ic:+.4f} (t={v.rank_ic_tstat:+.2f})")
    if v.mean_tertile_spread is not None:
        console.print(f"  top-minus-bottom tertile (21d fwd)={v.mean_tertile_spread:+.2%}")
    if v.deflated_sharpe is not None:
        dsr_mark = "✓" if v.passes_dsr else "✗"
        psr_mark = "✓" if v.passes_psr else "✗"
        console.print(
            f"  DSR={v.deflated_sharpe:.3f} {dsr_mark} (≥0.30)   "
            f"PSR={v.probabilistic_sharpe:.3f} {psr_mark} (≥0.70)"
        )
    verdict_color = "green" if v.passes else "yellow"
    console.print(f"  [{verdict_color}]{v.note}[/{verdict_color}]")
    console.print(
        "[dim]Survivorship-biased large-cap universe; deflated against the "
        "{composite, ridge, gbm} family. Research-only — promotes nothing.[/dim]"
    )


@forecast.command(
    "arima-eval",
    help="DSR/PSR-gated ARIMA conditional-mean (walk-forward, research-only; documents EMH).",
)
@click.option("--symbol", default="SPY", show_default=True)
def forecast_arima_eval(symbol: str) -> None:
    import pandas as pd

    from quant.data import bars
    from quant.forecast.arima import arima_research_verdict
    from quant.forecast.vol import log_returns

    settings = Settings()  # type: ignore[call-arg]
    path = bars._cache_path(symbol.upper(), settings.data_dir)
    if not path.exists():
        raise click.ClickException(
            f"No cached bars for {symbol.upper()} — run `quant data refresh`."
        )
    close = pd.read_parquet(path)["close"].dropna().to_numpy()
    returns = log_returns(close)
    v = arima_research_verdict(returns, d=0)
    console.print(
        f"[bold]{symbol.upper()} ARIMA conditional-mean OOS[/bold] — {v.n_oos} steps, "
        f"best ARIMA({v.best_p},0,{v.best_q})"
    )
    if v.mean_ic is not None:
        t = f" (t={v.ic_tstat:+.2f})" if v.ic_tstat is not None else ""
        console.print(f"  conditional IC={v.mean_ic:+.4f}{t}")
    if v.hit_rate is not None:
        console.print(f"  directional hit-rate={v.hit_rate:.1%}")
    if v.mse_ratio is not None:
        console.print(f"  MSE vs unconditional baseline={v.mse_ratio:.4f} (≥1 = no improvement)")
    if v.deflated_sharpe is not None:
        dsr_mark = "✓" if v.passes_dsr else "✗"
        psr_mark = "✓" if v.passes_psr else "✗"
        console.print(
            f"  DSR={v.deflated_sharpe:.3f} {dsr_mark} (≥0.30)   "
            f"PSR={v.probabilistic_sharpe:.3f} {psr_mark} (≥0.70)"
        )
    verdict_color = "green" if v.passes else "yellow"
    console.print(f"  [{verdict_color}]{v.note}[/{verdict_color}]")
    console.print(
        "[dim]Drift-neutral sign-strategy deflated against the (p,q) grid. Research-only "
        "— promotes nothing; a 'no edge' result is the documented, expected finding.[/dim]"
    )


@forecast.command(
    "regime", help="Live macro-conditioned regime read (HMM + credit cycle) + change-point."
)
def forecast_regime() -> None:
    from quant.forecast.regime import OOS_VERDICT, live_macro_regime, render_macro_regime

    settings = Settings()  # type: ignore[call-arg]
    r = live_macro_regime(settings, date.today(), oos_verdict=OOS_VERDICT)
    console.print(render_macro_regime(r))
    console.print(
        "[dim]RESEARCH ONLY — credit-conditioning's OOS IC gain was NOT robust (+0.037 at a cheap "
        "config -> ~0.000 at the production config); not wired into the analyst/MarketState. The "
        "existing market-only regime stands; the change-point detector is a separate validated tool. "
        "See `regime-eval`.[/dim]"
    )


@forecast.command(
    "regime-eval",
    help="Honest A/B: does adding the credit cycle improve OOS regime predictiveness?",
)
@click.option("--start", default="2000-01-01", show_default=True)
def forecast_regime_eval(start: str) -> None:
    from quant.forecast.regime import _load_macro_inputs, compare_regime_models

    settings = Settings()  # type: ignore[call-arg]
    s0 = pd.Timestamp(start).date()
    end = date.today()
    inp = _load_macro_inputs(settings.data_dir, s0, end)
    spy_close, vix = inp["spy_close"], inp["vix"]
    dgs10, dgs2, baa, aaa = inp["dgs10"], inp["dgs2"], inp["baa"], inp["aaa"]
    if (
        spy_close is None
        or vix is None
        or dgs10 is None
        or dgs2 is None
        or baa is None
        or aaa is None
    ):
        console.print(
            "[red]regime-eval: missing cached inputs (need SPY + VIX + DGS10/2 + BAA/AAA).[/red]"
        )
        return
    console.print("[dim]Refitting both HMMs walk-forward over the full history — ~1-2 min...[/dim]")
    cmp = compare_regime_models(
        spy_close=spy_close, vix=vix, dgs10=dgs10, dgs2=dgs2, baa=baa, aaa=aaa
    )
    for m in (cmp.market, cmp.macro):
        ic = f"{m.crisis_fwd_vol_ic:+.4f}" if m.crisis_fwd_vol_ic is not None else "n/a"
        sep = f"{m.fwd_vol_separation:+.3f}" if m.fwd_vol_separation is not None else "n/a"
        console.print(
            f"[bold]{m.name}[/bold] (feats={m.n_features}, n={m.n}): "
            f"crisis→fwd-vol IC={ic}, fwd-vol sep={sep}, "
            f"de-risk dd {m.dd_baseline:.1%}→{m.dd_derisked:.1%} ({m.dd_reduction:+.1%})"
        )
    if cmp.ic_improvement is not None:
        console.print(f"  IC improvement (macro minus market): {cmp.ic_improvement:+.4f}")
    console.print(f"[bold]VERDICT:[/bold] {cmp.verdict}")


@forecast.command(
    "changepoint", help="Bayesian online change-point detector (BOCPD) on SPY returns."
)
@click.option("--symbol", default="SPY", show_default=True)
def forecast_changepoint(symbol: str) -> None:
    import numpy as np

    from quant.data import bars
    from quant.forecast.regime import compute_change_points
    from quant.regime.features import _extract_close

    spy = bars.get_bars(
        bars.BarRequest(
            symbols=[symbol.upper()], start=date(date.today().year - 12, 1, 1), end=date.today()
        )
    )
    close = _extract_close(spy, symbol.upper())
    ret = pd.Series(np.log(close.astype(float).to_numpy()), index=close.index).diff().dropna()
    r = compute_change_points(ret)
    if r.cp_prob is None:
        console.print("Change-point: unavailable (insufficient history).")
        return
    console.print(
        f"[bold]{symbol.upper()} change-point[/bold] (asof {r.asof}): "
        f"cp_prob={r.cp_prob:.1%} (P run-length≤5), expected run={r.expected_run_length:.0f}d"
    )
    if r.recent_cp_dates:
        console.print("  recent break-probability spikes: " + ", ".join(r.recent_cp_dates))
    console.print(
        "[dim]Training-free, PIT online detector — flags SHARP breaks (e.g. COVID); a slow "
        "grind-bear (2022) is caught by the standing-regime HMM instead. Advisory only.[/dim]"
    )


def _regime_and_cp_series(settings: Settings, close: pd.Series) -> tuple[Any, Any]:
    """Build the macro-regime crisis-prob + change-point series for the ensemble (heavy)."""
    import numpy as np

    from quant.data import macro
    from quant.forecast.regime import (
        MacroRegimeConfig,
        build_macro_regime_features,
        change_point_series,
    )
    from quant.regime.detect import DetectConfig, run_detection

    feats = build_macro_regime_features(
        spy_close=close,
        vix=macro.get_series(macro.FRED_SERIES["vix"]),
        dgs10=macro.get_series(macro.FRED_SERIES["tenyear"]),
        dgs2=macro.get_series(macro.FRED_SERIES["twoyear"]),
        baa=macro.get_series(macro.FRED_SERIES["baa"]),
        aaa=macro.get_series(macro.FRED_SERIES["aaa"]),
        macro_config=MacroRegimeConfig(use_credit=True),
    )
    series = run_detection(
        feats, DetectConfig(refit_freq="QS", train_window_days=252 * 3, n_restarts=3)
    )
    ret = pd.Series(np.log(close.astype(float).to_numpy()), index=close.index).diff().dropna()
    cp = change_point_series(ret)["cp_prob"]
    return series["p_crisis"], cp


@forecast.command("ensemble", help="Live stacked forward-21d vol forecast (HAR+regime+cp). Slow.")
def forecast_ensemble() -> None:
    from quant.forecast.ensemble import live_stack, render_stack

    settings = Settings()  # type: ignore[call-arg]
    console.print("[dim]Fitting the stack (runs the regime walk-forward) — ~1-2 min...[/dim]")
    f = live_stack(settings, date.today())
    console.print(render_stack(f))
    console.print(
        "[dim]RESEARCH ONLY — the learned stack LOST to HAR-alone OOS (QLIKE +16%, DM p=0.003) "
        "and even to a naive average; the regime crisis-prob got ~0 weight. HAR stays the "
        "advisory vol champion; this is not wired into MarketState/analyst. See `ensemble-eval`.[/dim]"
    )


@forecast.command(
    "ensemble-eval",
    help="Honest nested purged walk-forward: does the stack beat HAR-alone + naive avg?",
)
@click.option("--start", default="2005-01-01", show_default=True)
def forecast_ensemble_eval(start: str) -> None:
    from quant.data import bars
    from quant.forecast.ensemble import StackConfig, build_base_panel, walk_forward_stack
    from quant.regime.features import _extract_close

    settings = Settings()  # type: ignore[call-arg]
    s0 = pd.Timestamp(start).date()
    spy = bars.get_bars(bars.BarRequest(symbols=["SPY"], start=s0, end=date.today()))
    close = _extract_close(spy, "SPY")
    console.print(
        "[dim]Running the regime walk-forward + nested purged stack eval — ~1-2 min...[/dim]"
    )
    p_crisis, cp = _regime_and_cp_series(settings, close)
    panel = build_base_panel(close, p_crisis=p_crisis, cp_prob=cp, config=StackConfig())
    ev = walk_forward_stack(panel, StackConfig())
    console.print(
        f"[bold]vol ensemble OOS[/bold] — {ev.n_oos} test points, best base = {ev.best_base}"
    )
    order = ("rw21", "ewma", "har", "regime", "cp", "eq3", "stack")
    console.print(
        "  mean QLIKE: "
        + ", ".join(f"{k}={ev.mean_qlike[k]:.4f}" for k in order if k in ev.mean_qlike)
    )
    if ev.avg_weights:
        console.print(
            "  avg learned weights: "
            + ", ".join(f"{k}={v:.2f}" for k, v in ev.avg_weights.items() if abs(v) > 1e-3)
        )
    if ev.dm_stack_vs_best:
        console.print(
            f"  DM stack-vs-{ev.best_base}: stat={ev.dm_stack_vs_best[0]:+.2f} "
            f"p={ev.dm_stack_vs_best[1]:.3f}  (stat<0 → stack better)"
        )
    console.print(f"[bold]VERDICT:[/bold] {ev.verdict}")


@cli.group(help="Market-wide regime detection (HMM/Kalman) — an observed, gated signal.")
def regime() -> None:
    pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _regime_series_path() -> Path:
    return Settings().data_dir / "regime" / "regime_series.parquet"  # type: ignore[call-arg]


@regime.command("fit", help="Refit the HMM walk-forward and persist the daily regime series.")
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
def regime_fit(start: str, end: str | None) -> None:
    from quant.regime.detect import (
        DetectConfig,
        fit_final_model,
        persist_model,
        persist_regime_series,
        run_detection,
    )
    from quant.regime.features import FeatureConfig, load_market_features
    from quant.research.registry import ExperimentRecord, append_experiment

    settings = Settings()  # type: ignore[call-arg]
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date() if end else pd.Timestamp.today().date()
    feats = load_market_features(start_date, end_date, FeatureConfig())
    series = run_detection(feats, DetectConfig())
    path = persist_regime_series(series, settings.data_dir)

    model_params, model_meta = fit_final_model(feats, DetectConfig())
    model_meta["git_sha"] = _git_sha()
    model_path = persist_model(model_params, model_meta, settings.data_dir)

    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"regime-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy="regime",
            kind="research",
            git_sha=_git_sha(),
            command=f"quant regime fit --start {start} --end {end_date}",
            params={"start": str(start_date), "end": str(end_date)},
            metrics={"n_days": float(len(series))},
            gates={},
            artifacts={"regime_series": str(path), "model": str(model_path)},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )
    console.print(f"[green]Wrote {len(series)} regime rows to {path}[/green]")
    console.print(f"[green]Wrote model to {model_path}[/green]")


@regime.command("label", help="Print the regime label + posterior as of a date (default latest).")
@click.option("--asof", default=None, help="Date (YYYY-MM-DD). Default: latest row.")
def regime_label(asof: str | None) -> None:
    path = _regime_series_path()
    if not path.exists():
        raise click.ClickException("No regime series. Run `quant regime fit` first.")
    frame = pd.read_parquet(path)
    row = frame.loc[pd.Timestamp(asof)] if asof else frame.iloc[-1]
    title_date = asof if asof else str(pd.Timestamp(frame.index[-1]).date())
    table = Table(title="Regime as of " + title_date)
    for col in ("Label", "p(calm)", "p(choppy)", "p(crisis)"):
        table.add_column(col)
    label_val = str(row["label"])
    p_calm = f"{float(row['p_calm']):.2f}"
    p_choppy = f"{float(row['p_choppy']):.2f}"
    p_crisis = f"{float(row['p_crisis']):.2f}"
    table.add_row(label_val, p_calm, p_choppy, p_crisis)
    console.print(table)


@regime.command("validate", help="Run the four out-of-sample gates and log to the registry.")
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default=None)
def regime_validate(start: str, end: str | None) -> None:
    from quant.data import bars
    from quant.regime.detect import DetectConfig, run_detection
    from quant.regime.features import FeatureConfig, _extract_close, load_market_features
    from quant.regime.validation import validate_regime_series
    from quant.research.registry import ExperimentRecord, append_experiment

    settings = Settings()  # type: ignore[call-arg]
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date() if end else pd.Timestamp.today().date()
    cfg = DetectConfig()
    feats = load_market_features(start_date, end_date, FeatureConfig())
    series = run_detection(feats, cfg)

    # Gate 4: real PIT check — labels invariant under a 90% truncation.
    cutoff = feats.index[int(len(feats) * 0.9)]
    trunc = run_detection(feats.loc[:cutoff], cfg)
    shared = trunc.index.intersection(series.index)
    pit_ok = bool((series.loc[shared, "label"] == trunc.loc[shared, "label"]).all())

    spy = bars.get_bars(bars.BarRequest(symbols=["SPY"], start=start_date, end=end_date))
    spy_ret = _extract_close(spy, "SPY").pct_change(fill_method=None)
    report = validate_regime_series(series, spy_returns=spy_ret, pit_consistent=pit_ok)
    gates = report.gates

    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"regime-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy="regime",
            kind="validation",
            git_sha=_git_sha(),
            command=f"quant regime validate --start {start} --end {end_date}",
            params={"start": str(start_date), "end": str(end_date)},
            metrics=report.metrics,
            gates=gates,
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )
    val_table = Table(title="Regime validation gates")
    val_table.add_column("Gate")
    val_table.add_column("Pass")
    for name, ok in gates.items():
        val_table.add_row(name, "[green]yes[/green]" if ok else "[red]no[/red]")
    console.print(val_table)
    console.print(f"Overall: {'PASS' if all(gates.values()) else 'FAIL'}")


@cli.group(help="Portfolio risk commands.")
def risk() -> None:
    pass


@risk.command("pretrade", help="Write a conservative pre-trade risk report.")
def risk_pretrade() -> None:
    from quant.live import run_rebalance
    from quant.risk.pretrade import build_pretrade_report

    settings = Settings()  # type: ignore[call-arg]
    rebalance_report = run_rebalance(
        dry_run=True,
        skip_safety_checks=True,
        record_bookkeeping=False,
    )
    orders = [order for outcome in rebalance_report.outcomes for order in outcome.orders]
    reference_prices: dict[str, float] = {}
    for outcome in rebalance_report.outcomes:
        reference_prices.update(outcome.reference_prices)

    report = build_pretrade_report(
        equity=rebalance_report.equity,
        orders=orders,
        reference_prices=reference_prices,
    )
    out = settings.data_dir / "risk" / "pretrade_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": report.passed,
                "equity": report.equity,
                "gross_exposure": report.gross_exposure,
                "symbol_weights": report.symbol_weights,
                "violations": [
                    {"code": v.code, "detail": v.detail, "symbol": v.symbol}
                    for v in report.violations
                ],
                "rebalance": {
                    "asof": rebalance_report.asof.isoformat(),
                    "dry_run": rebalance_report.dry_run,
                    "enabled_strategies": rebalance_report.enabled_strategies,
                    "total_orders": rebalance_report.total_orders,
                    "skipped_reason": rebalance_report.skipped_reason,
                    "outcomes": [
                        {
                            "strategy": outcome.slug,
                            "orders": len(outcome.orders),
                            "error": outcome.error,
                            "reference_prices": outcome.reference_prices,
                        }
                        for outcome in rebalance_report.outcomes
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "passed" if report.passed else "blocked"
    color = "green" if report.passed else "red"
    console.print(f"[{color}]Pre-trade risk {status}[/{color}]; wrote {out}")


@risk.command(
    "portfolio",
    help="Portfolio risk of the LIVE book: VaR/CVaR, vol, beta, exposure. Read-only analysis.",
)
@click.option("--lookback", default=180, show_default=True, type=int, help="Trading-day window.")
def risk_portfolio(lookback: int) -> None:
    from quant.risk.portfolio import live_portfolio_risk

    settings = Settings()  # type: ignore[call-arg]
    asof = date.today()
    positions: dict[str, int] = {}
    equity = 0.0
    try:
        client = AlpacaClient(settings=settings)
        equity = float(client.account().equity)
        positions = {p.symbol: int(p.qty) for p in client.positions()}
    except Exception as exc:
        raise click.ClickException(f"Alpaca unavailable: {exc!r}") from exc

    if not positions:
        console.print("[yellow]Book is flat — no portfolio risk to report.[/yellow]")
        return

    pr = live_portfolio_risk(positions, equity, asof=asof, lookback_days=lookback)
    if pr is None:
        console.print("[yellow]Could not compute portfolio risk (no bar history?).[/yellow]")
        return

    out = settings.data_dir / "risk" / "portfolio_risk.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "asof": asof.isoformat(),
                "equity": equity,
                "n_positions": pr.n_positions,
                "gross_exposure": pr.gross_exposure,
                "net_exposure": pr.net_exposure,
                "ann_vol": pr.ann_vol,
                "var_95": pr.var_95,
                "cvar_95": pr.cvar_95,
                "beta_to_benchmark": pr.beta_to_benchmark,
                "top_name_weight": pr.top_name_weight,
                "lookback_days": pr.lookback_days,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    console.rule(f"portfolio risk — {asof.isoformat()}")
    console.print(pr.render())
    console.print(f"[dim]wrote {out}[/dim]")


@risk.command(
    "scenarios",
    help="Stress the LIVE book under historical + hypothetical shock scenarios. Read-only.",
)
@click.option(
    "--lookback", default=180, show_default=True, type=int, help="Trading-day window for weights."
)
def risk_scenarios(lookback: int) -> None:
    from quant.risk.scenarios import live_stress

    settings = Settings()  # type: ignore[call-arg]
    asof = date.today()
    positions: dict[str, int] = {}
    equity = 0.0
    try:
        client = AlpacaClient(settings=settings)
        equity = float(client.account().equity)
        positions = {p.symbol: int(p.qty) for p in client.positions()}
    except Exception as exc:
        raise click.ClickException(f"Alpaca unavailable: {exc!r}") from exc

    if not positions:
        console.print("[yellow]Book is flat — no scenarios to stress.[/yellow]")
        return

    rep = live_stress(positions, equity, asof=asof, lookback_days=lookback)
    if rep is None or not rep.computable:
        console.print("[yellow]Could not compute stress (no bar history?).[/yellow]")
        return

    out = settings.data_dir / "risk" / f"scenarios.{asof.isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "asof": asof.isoformat(),
                "equity": equity,
                "worst_loss": rep.worst_loss,
                "worst_scenario": rep.worst_scenario,
                "degraded": list(rep.degraded),
                "results": [
                    {
                        "name": r.name,
                        "kind": r.kind,
                        "pnl_pct": r.pnl_pct,
                        "by_class": dict(r.by_class),
                        "missing_symbols": list(r.missing_symbols),
                        "computable": r.computable,
                    }
                    for r in rep.results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table = Table(title=f"stress scenarios — {asof.isoformat()}")
    table.add_column("Scenario")
    table.add_column("Kind")
    table.add_column("P&L", justify="right")
    for r in rep.results:
        pnl = "n/a" if r.pnl_pct is None else f"{r.pnl_pct:+.1%}"
        style = "red" if (r.pnl_pct is not None and r.pnl_pct < 0) else "green"
        marker = " ◀ worst" if r.name == rep.worst_scenario else ""
        table.add_row(r.name, r.kind, f"[{style}]{pnl}[/{style}]{marker}")
    console.print(table)
    console.print(rep.render())
    console.print(f"[dim]wrote {out}[/dim]")


@cli.group(
    help="Position sizing — an observed, comparison-only overlay (vol-target/Kelly/dd/regime)."
)
def sizing() -> None:
    pass


def _run_single_backtest(strategy_slug: str, start_date: date, end_date: date):  # type: ignore[no-untyped-def]
    """Run one default-param backtest and return its BacktestResult."""
    from quant.backtest.engine import run_backtest

    strategy_cls = REGISTRY[strategy_slug]
    universe = list(strategy_cls.spec.universe)
    bars = get_bars(BarRequest(symbols=universe, start=start_date, end=end_date))
    if bars.empty:
        raise click.ClickException(f"No bars for {strategy_slug!r} over {start_date}..{end_date}.")
    strat = strategy_cls.build(bars=bars)
    return run_backtest(strat, bars, BacktestConfig(), start_date, end_date)


def _load_regime_labels() -> pd.Series | None:
    path = _regime_series_path()
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "label" not in frame.columns:
        return None
    # sort_index: _as_of_label slices with .loc[:prior_ts], which requires a
    # monotonic index. The producer writes it sorted, but harden on read.
    return frame["label"].sort_index()


@sizing.command("compare", help="Compare a strategy's returns with vs without the sizing overlay.")
@click.argument("strategy")
@click.option("--start", default="2018-01-01", show_default=True)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
@click.option("--target-vol", default=0.15, show_default=True, type=float)
@click.option("--max-leverage", default=2.0, show_default=True, type=float)
@click.option("--kelly-fraction", default=0.5, show_default=True, type=float)
@click.option("--dd-floor", default=0.20, show_default=True, type=float)
@click.option("--no-vol-target", is_flag=True, default=False)
@click.option("--no-kelly", is_flag=True, default=False)
@click.option("--no-drawdown", is_flag=True, default=False)
@click.option("--no-regime", is_flag=True, default=False)
def sizing_compare(
    strategy: str,
    start: str,
    end: str | None,
    target_vol: float,
    max_leverage: float,
    kelly_fraction: float,
    dd_floor: float,
    no_vol_target: bool,
    no_kelly: bool,
    no_drawdown: bool,
    no_regime: bool,
) -> None:
    from quant.research.registry import ExperimentRecord, append_experiment
    from quant.sizing import SizingConfig, compare_sizing

    _require_strategy(strategy)
    settings = Settings()  # type: ignore[call-arg]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    console.print(f"[bold]Backtesting {strategy} {start_date}..{end_date}...[/bold]")
    result = _run_single_backtest(strategy, start_date, end_date)
    returns = result.returns
    if returns.empty:
        raise click.ClickException(f"Backtest for {strategy!r} produced no returns.")

    labels = _load_regime_labels()
    if labels is None:
        console.print("[yellow]No regime series found; regime component will be neutral.[/yellow]")

    config = SizingConfig(
        target_vol=target_vol,
        max_leverage=max_leverage,
        kelly_fraction=kelly_fraction,
        dd_floor=dd_floor,
        use_vol_target=not no_vol_target,
        use_kelly=not no_kelly,
        use_drawdown=not no_drawdown,
        use_regime=not no_regime,
    )
    comp = compare_sizing(returns, config, regime_labels=labels)

    table = Table(title=f"Sizing comparison — {strategy}")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Sized", justify="right")
    rows = [
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Max drawdown", "max_drawdown"),
        ("Ann vol", "ann_vol"),
        ("CAGR", "cagr"),
        ("Total return", "total_return"),
        ("Win rate", "win_rate"),
    ]
    for label_text, key in rows:
        table.add_row(label_text, f"{comp.baseline[key]:.4f}", f"{comp.sized[key]:.4f}")
    console.print(table)
    console.print(
        f"Gross exposure — mean {comp.gross_mean:.2f}, "
        f"min {comp.gross_min:.2f}, max {comp.gross_max:.2f}"
    )

    gates = {
        "gate_sharpe_improved": comp.sized["sharpe"] >= comp.baseline["sharpe"],
        "gate_maxdd_improved": comp.sized["max_drawdown"] >= comp.baseline["max_drawdown"],
    }
    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"sizing-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy=strategy,
            kind="research",
            git_sha=_git_sha(),
            command=f"quant sizing compare {strategy} --start {start_date} --end {end_date}",
            params={
                "target_vol": target_vol,
                "max_leverage": max_leverage,
                "kelly_fraction": kelly_fraction,
                "dd_floor": dd_floor,
                "use_vol_target": not no_vol_target,
                "use_kelly": not no_kelly,
                "use_drawdown": not no_drawdown,
                "use_regime": not no_regime,
            },
            metrics={f"sized_{k}": v for k, v in comp.sized.items()}
            | {"gross_mean": comp.gross_mean},
            gates=gates,
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )


@cli.group(help="Options/Greeks engine + protective hedging overlay — observed-only.")
def hedge() -> None:
    pass


@hedge.command(
    "surface", help="Live SPY implied-vol surface: ATM IV, term structure, put skew (read-only)."
)
def hedge_surface() -> None:
    from quant.options.surface import live_vol_surface, render_vol_surface

    settings = Settings()  # type: ignore[call-arg]
    v = live_vol_surface(settings, date.today())
    console.print(render_vol_surface(v))
    console.print(
        f"  spot={v.spot}  near={v.near_dte}d ATM_IV={v.atm_iv_30d}  far={v.far_dte}d "
        f"ATM_IV={v.atm_iv_90d}  quotes={v.n_quotes}/{v.n_expiries}exp"
    )


def _spy_close_series(spy_bars: pd.DataFrame) -> pd.Series:
    """Extract a clean SPY close series from a get_bars (symbol, field) frame."""
    if isinstance(spy_bars.columns, pd.MultiIndex):
        close = spy_bars["SPY"]["close"]
    else:
        close = spy_bars["close"] if "close" in spy_bars.columns else spy_bars.iloc[:, 0]
    out: pd.Series = close.sort_index().astype(float)
    return out


@hedge.command("price", help="Black-Scholes price + Greeks (and implied vol if --mark given).")
@click.option("--spot", required=True, type=float)
@click.option("--strike", required=True, type=float)
@click.option("--days", required=True, type=float, help="Calendar days to expiry.")
@click.option("--vol", default=0.20, show_default=True, type=float, help="Annualized vol.")
@click.option("--right", default="put", show_default=True, type=click.Choice(["put", "call"]))
@click.option("--rate", default=0.03, show_default=True, type=float)
@click.option("--div", default=0.015, show_default=True, type=float)
@click.option("--mark", default=None, type=float, help="Market price -> solve implied vol.")
def hedge_price(
    spot: float,
    strike: float,
    days: float,
    vol: float,
    right: str,
    rate: float,
    div: float,
    mark: float | None,
) -> None:
    from quant.options import bs_greeks, bs_price, implied_vol

    t_years = days / 365.0
    price = bs_price(spot, strike, t_years, vol, rate, div, right)
    g = bs_greeks(spot, strike, t_years, vol, rate, div, right)
    table = Table(title=f"{right.capitalize()} {strike:g} / {days:g}d on spot {spot:g}")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("price", f"{price:.4f}")
    table.add_row("delta", f"{g.delta:.4f}")
    table.add_row("gamma", f"{g.gamma:.6f}")
    table.add_row("vega", f"{g.vega:.4f}")
    table.add_row("theta", f"{g.theta:.4f}")
    table.add_row("rho", f"{g.rho:.4f}")
    if mark is not None:
        iv = implied_vol(mark, spot, strike, t_years, rate, div, right)
        table.add_row("implied vol", f"{iv:.4f}")
    console.print(table)


@hedge.command(
    "compare", help="Compare a strategy's returns with vs without the SPY hedge overlay."
)
@click.argument("strategy")
@click.option("--start", default="2018-01-01", show_default=True)
@click.option("--end", default=None, help="History end (YYYY-MM-DD). Default: today.")
@click.option(
    "--structure",
    default="put",
    show_default=True,
    type=click.Choice(["put", "collar", "put_spread"]),
)
@click.option("--put-moneyness", default=0.05, show_default=True, type=float)
@click.option("--call-moneyness", default=0.05, show_default=True, type=float)
@click.option("--spread-width", default=0.10, show_default=True, type=float)
@click.option("--coverage", default=1.0, show_default=True, type=float)
@click.option("--tenor-days", default=30, show_default=True, type=int)
@click.option("--roll-days", default=21, show_default=True, type=int)
@click.option("--no-regime", is_flag=True, default=False)
def hedge_compare(
    strategy: str,
    start: str,
    end: str | None,
    structure: str,
    put_moneyness: float,
    call_moneyness: float,
    spread_width: float,
    coverage: float,
    tenor_days: int,
    roll_days: int,
    no_regime: bool,
) -> None:
    from quant.options import HedgeConfig, compare_hedge
    from quant.research.registry import ExperimentRecord, append_experiment

    _require_strategy(strategy)
    settings = Settings()  # type: ignore[call-arg]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    console.print(f"[bold]Backtesting {strategy} {start_date}..{end_date}...[/bold]")
    result = _run_single_backtest(strategy, start_date, end_date)
    returns = result.returns
    if returns.empty:
        raise click.ClickException(f"Backtest for {strategy!r} produced no returns.")

    spy_bars = get_bars(BarRequest(symbols=["SPY"], start=start_date, end=end_date))
    if spy_bars.empty:
        raise click.ClickException("No SPY bars cached for the hedge underlying.")
    spy_close = _spy_close_series(spy_bars)

    labels = _load_regime_labels()
    if labels is None and not no_regime:
        console.print("[yellow]No regime series found; hedge intensity will be neutral.[/yellow]")

    config = HedgeConfig(
        structure=structure,
        put_moneyness=put_moneyness,
        call_moneyness=call_moneyness,
        spread_width=spread_width,
        coverage=coverage,
        tenor_days=tenor_days,
        roll_days=roll_days,
        use_regime=not no_regime,
    )
    comp = compare_hedge(returns, spy_close, config, regime_labels=labels)

    table = Table(title=f"Hedge comparison — {strategy} ({structure})")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Hedged", justify="right")
    rows = [
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Max drawdown", "max_drawdown"),
        ("CVaR 5%", "cvar_5"),
        ("Worst day", "worst_day"),
        ("Ann vol", "ann_vol"),
        ("CAGR", "cagr"),
        ("Total return", "total_return"),
    ]
    for label_text, key in rows:
        table.add_row(label_text, f"{comp.baseline[key]:.4f}", f"{comp.hedged[key]:.4f}")
    console.print(table)
    console.print(
        f"Hedge cost — {comp.n_rolls} rolls, total premium {comp.total_premium:.4f}, "
        f"~{comp.premium_drag_annual:.4f}/yr, mean contracts {comp.mean_contracts:.4f}"
    )

    gates = {
        "gate_maxdd_improved": comp.hedged["max_drawdown"] >= comp.baseline["max_drawdown"],
        "gate_cvar_improved": comp.hedged["cvar_5"] >= comp.baseline["cvar_5"],
    }
    append_experiment(
        settings.data_dir / "research" / "experiments.jsonl",
        ExperimentRecord(
            run_id=f"hedge-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(UTC).replace(microsecond=0),
            strategy=strategy,
            kind="research",
            git_sha=_git_sha(),
            command=f"quant hedge compare {strategy} --structure {structure}",
            params={
                "structure": structure,
                "put_moneyness": put_moneyness,
                "call_moneyness": call_moneyness,
                "spread_width": spread_width,
                "coverage": coverage,
                "tenor_days": tenor_days,
                "roll_days": roll_days,
                "use_regime": not no_regime,
            },
            metrics={f"hedged_{k}": v for k, v in comp.hedged.items()}
            | {
                "total_premium": comp.total_premium,
                "premium_drag_annual": comp.premium_drag_annual,
            },
            gates=gates,
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=0.0,
        ),
    )


@cli.command(help="List all registered strategies.")
def strategies() -> None:
    settings = (
        Settings.model_construct(data_dir=Path("./data"))
        if not _can_load_settings()
        else Settings()  # type: ignore[call-arg]
    )
    governance_labels = _governance_state_labels(settings.data_dir)

    table = Table(title="Registered strategies", show_header=True)
    table.add_column("Slug")
    table.add_column("Name")
    table.add_column("Rebalance")
    table.add_column("Universe size", justify="right")
    table.add_column("Live", justify="center")
    table.add_column("Governance", justify="center")
    for spec in list_strategies():
        table.add_row(
            spec.slug,
            spec.name,
            spec.rebalance_frequency,
            str(len(spec.universe)),
            "yes" if spec.enabled_live else "no",
            governance_labels.get(spec.slug, "unknown"),
        )
    console.print(table)


@cli.group(help="Monitoring daemon (guardrails + auto kill-switch) — can HALT, never resumes.")
def guard() -> None:
    pass


def _enabled_universe_symbols() -> list[str]:
    """Union of the live-enabled strategies' universes (for bar-freshness)."""
    from quant.live.safety import enabled_strategy_slugs

    symbols: set[str] = set()
    for slug in enabled_strategy_slugs():
        symbols.update(REGISTRY[slug].spec.universe)
    return sorted(symbols)


def _best_effort_positions(settings: Settings) -> tuple[list[Any] | None, str]:
    """Fetch Alpaca positions for reconciliation; (None, note) on any failure."""
    try:
        client = AlpacaClient(settings=settings)
        return list(client.positions()), "ok"
    except Exception as exc:  # recon is optional; degrade gracefully
        return None, f"alpaca unavailable: {exc!r}"


def _best_effort_equity(settings: Settings) -> float | None:
    """Fetch the broker's authoritative account equity; None on any failure.

    Feeds the guard's equity-health guardrail so a healthy flat book ($1M, no
    positions) is distinguishable from a dead local equity feed."""
    try:
        return float(AlpacaClient(settings=settings).account().equity)
    except Exception:  # equity read is best-effort; the guardrail handles None
        return None


def _best_effort_news(settings: Settings) -> Any:
    """Recent-news sentiment for the analyst context; empty read on any failure."""
    try:
        from quant.nlp.sentiment import live_news_sentiment

        return live_news_sentiment(settings)
    except Exception:  # news is optional context
        return None


def _best_effort_event_risk(settings: Settings, asof: date) -> Any:
    """Macro/policy/event-risk read for the analyst context; None on any failure."""
    try:
        from quant.macro.events import live_event_risk

        return live_event_risk(settings, asof)
    except Exception:  # event risk is optional context
        return None


def _best_effort_fundamentals(settings: Settings, asof: date) -> Any:
    """Cross-sectional fundamentals read for the analyst context; None on failure."""
    try:
        from quant.fundamentals.factors import live_fundamentals

        return live_fundamentals(settings, asof)
    except Exception:  # fundamentals are optional context
        return None


def _best_effort_nowcast(settings: Settings, asof: date) -> Any:
    """Macro / business-cycle nowcast for the analyst context; None on failure."""
    try:
        from quant.macro.nowcast import live_macro_nowcast

        return live_macro_nowcast(settings, asof)
    except Exception:  # the nowcast is optional context
        return None


def _best_effort_vol_surface(settings: Settings, asof: date) -> Any:
    """Implied-vol surface (IV/term/skew) for the analyst context; None on failure."""
    try:
        from quant.options.surface import live_vol_surface

        return live_vol_surface(settings, asof)
    except Exception:  # the vol surface is optional context
        return None


def _best_effort_vol_forecast(settings: Settings, asof: date) -> Any:
    """HAR-RV vol forecast (validated OOS) for the analyst context; None on failure."""
    try:
        from quant.forecast.vol import OOS_SKILL_SPY, live_vol_forecast

        return live_vol_forecast(settings, asof, symbol="SPY", oos_skill=OOS_SKILL_SPY)
    except Exception:  # the forecast is optional context
        return None


def _render_guardrail_table(report: Any) -> Table:
    table = Table(title="Guardrails", show_header=True)
    table.add_column("Guardrail")
    table.add_column("Severity")
    table.add_column("Detail")
    palette = {"ok": "green", "warn": "yellow", "halt": "red"}
    for o in report.outcomes:
        color = palette.get(o.severity, "white")
        table.add_row(o.name, f"[{color}]{o.severity}[/{color}]", o.detail)
    return table


@guard.command(
    "check", help="Evaluate guardrails once and print. Never halts, never writes status."
)
def guard_check() -> None:
    from quant.monitor.daemon import format_heartbeat, gather_inputs
    from quant.monitor.guardrails import GuardrailConfig, evaluate_guardrails

    settings = Settings()  # type: ignore[call-arg]
    config = GuardrailConfig()
    positions, note = _best_effort_positions(settings)
    if positions is None:
        console.print(f"[yellow]reconciliation skipped — {note}[/yellow]")
    inputs = gather_inputs(
        settings.data_dir,
        asof=date.today(),
        config=config,
        alpaca_positions=positions,
        live_equity=_best_effort_equity(settings),
        symbols=_enabled_universe_symbols(),
    )
    report = evaluate_guardrails(inputs, config)
    console.print(_render_guardrail_table(report))
    hb = format_heartbeat(
        inputs, report, datetime.now(UTC).replace(microsecond=0), halt_active=False
    )
    console.print(hb)
    if report.halting:
        console.print(
            "[red]A halt-severity guardrail is tripped. `quant guard run` would halt trading.[/red]"
        )


@guard.command(
    "run", help="Run the monitoring daemon. Auto-halts on a halt verdict (unless --dry-run)."
)
@click.option(
    "--interval", default=300.0, show_default=True, type=float, help="Seconds between ticks."
)
@click.option("--once", is_flag=True, default=False, help="Run a single tick and exit.")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Evaluate + report but never set the halt."
)
@click.option("--max-ticks", default=None, type=int, help="Stop after N ticks (default: forever).")
def guard_run(interval: float, once: bool, dry_run: bool, max_ticks: int | None) -> None:
    from quant.monitor.daemon import run_loop, run_once
    from quant.monitor.guardrails import GuardrailConfig

    settings = Settings()  # type: ignore[call-arg]
    config = GuardrailConfig()
    symbols = _enabled_universe_symbols()

    def positions_fn() -> list[Any] | None:
        pos, _ = _best_effort_positions(settings)
        return pos

    def live_equity_fn() -> float | None:
        return _best_effort_equity(settings)

    if once:
        res = run_once(
            settings.data_dir,
            config,
            alpaca_positions=positions_fn(),
            live_equity=live_equity_fn(),
            symbols=symbols,
            dry_run=dry_run,
        )
        console.print(res.heartbeat)
        if res.halt_triggered:
            console.print(
                "[bold red]TRADING HALTED by the monitor. "
                "Investigate, then resume with `quant governance resume --reason ...`.[/bold red]"
            )
        return

    console.print(
        f"[bold]Monitor daemon starting (interval={interval}s, dry_run={dry_run}). "
        "Ctrl-C to stop. The daemon can HALT but never resumes.[/bold]"
    )
    from quant.deploy.alerts import AlertClient, AlertConfig

    _alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )
    )
    # Box-recovery signal: the guard is KeepAlive, so it (re)starts on boot/crash.
    # A one-line Slack ping on start is the positive counterpart to the off-box
    # dead-man's-switch — "the box is back and the safety daemon is live."
    _alerts.send_slack(
        f":white_check_mark: quant guard online ({'dry-run' if dry_run else 'LIVE'}) — monitoring resumed."
    )
    run_loop(
        settings.data_dir,
        config,
        interval_s=interval,
        dry_run=dry_run,
        max_ticks=max_ticks,
        alpaca_positions_fn=positions_fn,
        live_equity_fn=live_equity_fn,
        symbols=symbols,
        console_print=lambda s: console.print(s),
        heartbeat_ping=lambda: _alerts.ping_success(settings.healthcheck_guard_url),
    )


@cli.group(help="Deployment ops: the local tick scheduler (M4 host).")
def ops() -> None:
    pass


@ops.command("tick", help="One scheduler tick: run any due jobs. Called by launchd every 60s.")
def ops_tick() -> None:
    from quant.deploy.alerts import AlertClient, AlertConfig
    from quant.deploy.dispatcher import Dispatcher
    from quant.deploy.manifest import load_manifest
    from quant.governance.halt import load_halt

    settings = Settings()  # type: ignore[call-arg]
    alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )
    )
    manifest_path = Path(__file__).resolve().parent / "deploy" / "jobs.toml"
    disp = Dispatcher(
        data_dir=settings.data_dir,
        manifest=load_manifest(manifest_path),
        alerts=alerts,
        halt_active=lambda: load_halt(settings.data_dir).active,
        tick_url=settings.healthcheck_tick_url,
    )
    raise SystemExit(disp.tick())


@ops.command("run-job", help="Manually run one manifest job now (recovery for MISSED_CRITICAL).")
@click.argument("name")
@click.option("--force", is_flag=True, help="Run even if outside the window / already marked.")
def ops_run_job(name: str, force: bool) -> None:
    from quant.deploy.dispatcher import Dispatcher, _expand
    from quant.deploy.manifest import load_manifest

    settings = Settings()  # type: ignore[call-arg]
    manifest = load_manifest(Path(__file__).resolve().parent / "deploy" / "jobs.toml")
    job = next((j for j in manifest.jobs if j.name == name), None)
    if job is None:
        raise click.ClickException(f"unknown job: {name}")
    if not force:
        raise click.ClickException("refusing to run off-schedule without --force")
    disp = Dispatcher(data_dir=settings.data_dir, manifest=manifest)
    for args in _expand(job):
        rc = disp.runner(args, Path(__file__).resolve().parents[1])
        if rc != 0:
            raise SystemExit(rc)


@cli.group(help="Analyst layer — a daily Claude-written digest delivered to Slack (read-only).")
def analyst() -> None:
    pass


@analyst.command(
    "spend",
    help="Claude-spend meter: total + per-day / model / call-site cost from the ledger. Read-only.",
)
@click.option(
    "--asof", default=None, help="Highlight this day's spend YYYY-MM-DD (default: today)."
)
@click.option("--days", type=int, default=14, help="Recent days to show in the by-day table.")
def analyst_spend(asof: str | None, days: int) -> None:
    from rich.table import Table

    from quant.analyst.spend import ledger_path, load_records, summarize

    settings = Settings()  # type: ignore[call-arg]
    asof_date = (date.fromisoformat(asof) if asof else date.today()).isoformat()
    records = load_records(settings.data_dir)
    s = summarize(records, asof_date=asof_date)

    console.rule(f"Claude spend — {s['calls']} calls, ${s['total_usd']:.4f} total")
    console.print(
        f"today ({asof_date}): ${(s['today_usd'] or 0.0):.4f}    ledger: "
        f"{ledger_path(settings.data_dir)}"
    )
    if not records:
        console.print("[yellow]no spend recorded yet[/yellow]")
        return

    for title, key, label in (
        (f"by day (last {days})", "by_day", "date"),
        ("by model", "by_model", "model"),
        ("by call-site", "by_call_site", "call_site"),
    ):
        table = Table(title=title)
        table.add_column(label)
        table.add_column("USD", justify="right")
        items = list(s[key].items())
        if key == "by_day":
            items = items[-days:]
        for name, val in items:
            table.add_row(str(name), f"${val:.4f}")
        console.print(table)


@analyst.command(
    "digest", help="Build today's digest and post it to Slack. Read-only — never trades or halts."
)
@click.option("--asof", default=None, help="Session date YYYY-MM-DD (default: today).")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print the digest; do not post to Slack."
)
def analyst_digest(asof: str | None, dry_run: bool) -> None:
    from quant.analyst import run_digest
    from quant.deploy.alerts import AlertClient, AlertConfig
    from quant.governance.halt import load_halt

    settings = Settings()  # type: ignore[call-arg]
    session_date = date.fromisoformat(asof) if asof else date.today()

    alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )
    )

    # Best-effort live snapshot from Alpaca; the digest degrades gracefully without it.
    account: dict[str, float] | None = None
    live_positions: list[tuple[str, int]] | None = None
    try:
        client = AlpacaClient(settings=settings)
        acct = client.account()
        account = {"equity": acct.equity, "last_equity": acct.last_equity, "cash": acct.cash}
        live_positions = [(p.symbol, int(p.qty)) for p in client.positions()]
    except Exception as exc:  # digest is best-effort — a broker hiccup must not fail it
        console.print(f"[yellow]analyst: Alpaca snapshot unavailable — {exc!r}[/yellow]")

    governance_live, _ = _doctor_governance_live_slugs(settings.data_dir)
    artifact_dir = Path(__file__).resolve().parents[1] / "docs" / "analyst"
    result = run_digest(
        data_dir=settings.data_dir,
        asof=session_date,
        settings=settings,
        alerts=alerts,
        artifact_dir=artifact_dir,
        dry_run=dry_run,
        account=account,
        live_positions=live_positions,
        governance_live=governance_live,
        halt_active=load_halt(settings.data_dir).active,
    )

    console.rule(f"analyst digest — {session_date.isoformat()}")
    console.print(result.body)
    console.rule()
    src = "Claude" if result.used_llm else "template (no ANTHROPIC_API_KEY or call failed)"
    if dry_run:
        where = "DRY-RUN (not sent)"
    elif result.delivered:
        where = "posted to Slack"
    else:
        where = "Slack not configured / post failed"
    console.print(f"[dim]source: {src} · {where} · artifact: {result.artifact_path}[/dim]")


@analyst.command(
    "brief",
    help="Claude decision-maker (Phase A): richer read-only context → structured brief → Slack. "
    "Advisory only — places no orders, sets no halt, changes no allocation.",
)
@click.option("--asof", default=None, help="Session date YYYY-MM-DD (default: today).")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print the brief; do not post to Slack."
)
def analyst_brief(asof: str | None, dry_run: bool) -> None:
    from quant.analyst import gather_digest_data, render_facts
    from quant.analyst.advisor import advise
    from quant.analyst.context import gather_analyst_context, render_context
    from quant.deploy.alerts import AlertClient, AlertConfig
    from quant.governance.halt import load_halt

    settings = Settings()  # type: ignore[call-arg]
    session_date = date.fromisoformat(asof) if asof else date.today()
    alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )
    )

    # Best-effort live Alpaca snapshot; everything degrades gracefully without it.
    account: dict[str, float] | None = None
    live_positions: list[tuple[str, int]] | None = None
    try:
        client = AlpacaClient(settings=settings)
        acct = client.account()
        account = {"equity": acct.equity, "last_equity": acct.last_equity, "cash": acct.cash}
        live_positions = [(p.symbol, int(p.qty)) for p in client.positions()]
    except Exception as exc:  # best-effort — a broker hiccup must not fail the brief
        console.print(f"[yellow]analyst: Alpaca snapshot unavailable — {exc!r}[/yellow]")

    governance_live, _ = _doctor_governance_live_slugs(settings.data_dir)
    data = gather_digest_data(
        settings.data_dir,
        session_date,
        dry_run=dry_run,
        account=account,
        live_positions=live_positions,
        governance_live=governance_live,
        halt_active=load_halt(settings.data_dir).active,
    )
    facts = render_facts(data)
    positions_dict = dict(live_positions) if live_positions else None
    equity_val = account.get("equity") if account else None
    ctx = gather_analyst_context(
        settings.data_dir,
        session_date,
        positions=positions_dict,
        equity=equity_val,
        news=_best_effort_news(settings),
        event_risk=_best_effort_event_risk(settings, session_date),
        fundamentals=_best_effort_fundamentals(settings, session_date),
        macro_nowcast=_best_effort_nowcast(settings, session_date),
        vol_surface=_best_effort_vol_surface(settings, session_date),
        vol_forecast=_best_effort_vol_forecast(settings, session_date),
    )
    context_text = render_context(ctx)

    brief = advise(
        facts,
        context_text,
        settings=settings,
        asof=session_date,
        data_dir=settings.data_dir,
    )
    body = brief.render() if brief is not None else facts

    artifact_dir = Path(__file__).resolve().parents[1] / "docs" / "analyst"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"brief-{session_date.isoformat()}.md"
    src_label = "Claude (Phase A advisor)" if brief is not None else "context + facts (no LLM)"
    artifact.write_text(
        f"# Analyst brief — {session_date.isoformat()}\n\n{body}\n\n"
        f"---\n\n**Context**\n\n```\n{context_text}\n```\n\n"
        f"**Facts**\n\n```\n{facts}\n```\n\n_Source: {src_label}. Advisory only — nothing applied._\n",
        encoding="utf-8",
    )

    delivered = False
    if not dry_run:
        text = f"🧭 quant analyst brief — {session_date.isoformat()}\n\n{body}"
        delivered = alerts.send_slack(text)

    console.rule(f"analyst brief — {session_date.isoformat()}")
    console.print(body)
    console.rule()
    where = "DRY-RUN (not sent)" if dry_run else ("posted to Slack" if delivered else "not sent")
    console.print(f"[dim]source: {src_label} · {where} · artifact: {artifact}[/dim]")


@analyst.command(
    "watch",
    help="Intraday Claude watch: a short READ-ONLY commentary on the live book, posted to "
    "Slack (slots: open/midday/power-hour). Advisory only — places no orders, sets no halt, "
    "changes no allocation; bounded + non-spammy.",
)
@click.option("--asof", default=None, help="Session date YYYY-MM-DD (default: today).")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print the note; do not post to Slack."
)
@click.option("--slot", default="midday", help="Intraday slot label (e.g. open/midday/power-hour).")
def analyst_watch(asof: str | None, dry_run: bool, slot: str) -> None:
    from quant.analyst import run_watch
    from quant.analyst.context import gather_analyst_context, render_context
    from quant.deploy.alerts import AlertClient, AlertConfig
    from quant.governance.halt import load_halt

    settings = Settings()  # type: ignore[call-arg]
    session_date = date.fromisoformat(asof) if asof else date.today()
    alerts = AlertClient(
        AlertConfig(
            healthcheck_tick_url=settings.healthcheck_tick_url,
            healthcheck_guard_url=settings.healthcheck_guard_url,
            pushover_app_token=settings.pushover_app_token,
            pushover_user_key=settings.pushover_user_key,
            slack_webhook_url=settings.slack_webhook_url,
        )
    )

    # Best-effort live Alpaca snapshot; everything degrades gracefully without it.
    account: dict[str, float] | None = None
    live_positions: list[tuple[str, int]] | None = None
    try:
        client = AlpacaClient(settings=settings)
        acct = client.account()
        account = {"equity": acct.equity, "last_equity": acct.last_equity, "cash": acct.cash}
        live_positions = [(p.symbol, int(p.qty)) for p in client.positions()]
    except Exception as exc:  # best-effort — a broker hiccup must not fail the watch
        console.print(f"[yellow]watch: Alpaca snapshot unavailable — {exc!r}[/yellow]")

    governance_live, _ = _doctor_governance_live_slugs(settings.data_dir)
    positions_dict = dict(live_positions) if live_positions else None
    equity_val = account.get("equity") if account else None
    ctx = gather_analyst_context(
        settings.data_dir,
        session_date,
        positions=positions_dict,
        equity=equity_val,
        news=_best_effort_news(settings),
        event_risk=_best_effort_event_risk(settings, session_date),
        fundamentals=_best_effort_fundamentals(settings, session_date),
        macro_nowcast=_best_effort_nowcast(settings, session_date),
        vol_surface=_best_effort_vol_surface(settings, session_date),
        vol_forecast=_best_effort_vol_forecast(settings, session_date),
    )
    context_text = render_context(ctx)

    result = run_watch(
        data_dir=settings.data_dir,
        asof=session_date,
        settings=settings,
        alerts=alerts,
        slot=slot,
        dry_run=dry_run,
        account=account,
        live_positions=live_positions,
        governance_live=governance_live,
        halt_active=load_halt(settings.data_dir).active,
        context_text=context_text,
    )

    console.rule(f"analyst watch [{slot}] — {session_date.isoformat()}")
    console.print(result.body or "(suppressed — nothing posted)")
    console.rule()
    src = "Claude" if result.used_llm else "template (no LLM)"
    if result.suppressed_reason:
        where = f"suppressed: {result.suppressed_reason}"
    elif dry_run:
        where = "DRY-RUN (not sent)"
    else:
        where = "posted to Slack" if result.posted else "not sent"
    console.print(f"[dim]source: {src} · {where} · advisory only — nothing applied[/dim]")


@analyst.command(
    "propose",
    help="Claude decision-maker Phase B: structured advisory proposals (de-risk throttle, "
    "allocation tilts, halt recommendation), governance-clamped and logged. Applies NOTHING.",
)
@click.option("--asof", default=None, help="Session date YYYY-MM-DD (default: today).")
def analyst_propose(asof: str | None) -> None:
    from quant.analyst import gather_digest_data, render_facts
    from quant.analyst.advisor import propose
    from quant.analyst.context import gather_analyst_context, render_context
    from quant.governance.halt import load_halt

    settings = Settings()  # type: ignore[call-arg]
    session_date = date.fromisoformat(asof) if asof else date.today()

    account: dict[str, float] | None = None
    live_positions: list[tuple[str, int]] | None = None
    try:
        client = AlpacaClient(settings=settings)
        acct = client.account()
        account = {"equity": acct.equity, "last_equity": acct.last_equity, "cash": acct.cash}
        live_positions = [(p.symbol, int(p.qty)) for p in client.positions()]
    except Exception as exc:  # best-effort
        console.print(f"[yellow]analyst: Alpaca snapshot unavailable — {exc!r}[/yellow]")

    governance_live, _ = _doctor_governance_live_slugs(settings.data_dir)
    data = gather_digest_data(
        settings.data_dir,
        session_date,
        account=account,
        live_positions=live_positions,
        governance_live=governance_live,
        halt_active=load_halt(settings.data_dir).active,
    )
    facts = render_facts(data)
    positions_dict = dict(live_positions) if live_positions else None
    equity_val = account.get("equity") if account else None
    ctx = gather_analyst_context(
        settings.data_dir,
        session_date,
        positions=positions_dict,
        equity=equity_val,
        news=_best_effort_news(settings),
        event_risk=_best_effort_event_risk(settings, session_date),
        fundamentals=_best_effort_fundamentals(settings, session_date),
        macro_nowcast=_best_effort_nowcast(settings, session_date),
        vol_surface=_best_effort_vol_surface(settings, session_date),
        vol_forecast=_best_effort_vol_forecast(settings, session_date),
    )
    context_text = render_context(ctx)

    proposals = propose(
        facts,
        context_text,
        settings=settings,
        asof=session_date,
        live_slugs=governance_live,
        data_dir=settings.data_dir,
    )
    console.rule(f"analyst proposals (advise-and-log) — {session_date.isoformat()}")
    if proposals is not None:
        console.print(proposals.render())
        console.print(
            "[dim]Phase B: clamped by governance, logged to data/analyst/decisions.jsonl, "
            "applied to NOTHING.[/dim]"
        )
    else:
        console.print("[yellow](no proposals — no ANTHROPIC_API_KEY or the call failed)[/yellow]")


if __name__ == "__main__":  # pragma: no cover
    cli()
