"""Click CLI: top-level group + every subcommand wired to the strategy registry.

Foundation phase: most subcommands are stubs that raise `click.ClickException`
with a clear "not yet implemented in Plan N" message. `status` and `data` are
fully functional; the rest are scaffolded so the command surface is stable.
"""

from __future__ import annotations

import webbrowser
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

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

    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(),
        start=start_date,
        end=end_date,
    )

    table = Table(title="Combined-book result", show_header=True)
    table.add_column("Strategy")
    table.add_column("Alloc", justify="right")
    table.add_column("End Equity", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Max DD", justify="right")
    for slug in sorted(result.per_strategy):
        sub = result.per_strategy[slug]
        table.add_row(
            slug,
            f"{result.allocation.get(slug, 0):.1%}",
            f"${sub.ending_equity:,.0f}",
            f"{sharpe(sub.returns):.2f}",
            f"{cagr(sub.returns):.2%}",
            f"{max_drawdown(sub.returns):.2%}",
        )
    table.add_section()
    table.add_row(
        "[bold]COMBINED[/]",
        "100.0%",
        f"${result.ending_equity:,.0f}",
        f"{sharpe(result.returns):.2f}",
        f"{cagr(result.returns):.2%}",
        f"{max_drawdown(result.returns):.2%}",
    )
    console.print(table)

    from quant.backtest import write_combined_tearsheet

    settings = Settings()  # type: ignore[call-arg]
    out_dir = settings.data_dir / "backtests" / "_combined"
    html_path = write_combined_tearsheet(result=result, out_dir=out_dir)
    console.print(f"[green]Wrote {html_path}[/green]")


def _write_validation_report_json(
    *,
    out_dir: Path,
    slug: str,
    run_date: date,
    data_start: date,
    data_end: date,
    report: Any,
    provenance: str,
) -> Path:
    import json

    n_tested = sum(1 for r in report.regime_breakdown if r.n_days >= 30)
    payload = {
        "slug": slug,
        "run_date": run_date.isoformat(),
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
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
    cpcv_groups: int,
    cpcv_k_test: int,
    quick: bool,
    holdout_years: int,
) -> None:
    from datetime import timedelta as _td

    from quant.backtest.cpcv import CPCVConfig
    from quant.backtest.validation import run_validation

    _require_strategy(strategy)

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
    )
    validation_json = _write_validation_report_json(
        out_dir=out_dir,
        slug=strategy,
        run_date=date.today(),
        data_start=start_date,
        data_end=end_date,
        report=report,
        provenance=f"quant validate {strategy} --start {start_date} --end {end_date}",
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
def rebalance(dry_run: bool, asof: str | None, strategy_filter: str | None) -> None:
    from quant.live import run_rebalance

    asof_date = date.fromisoformat(asof) if asof else date.today()
    strategies_arg = [strategy_filter] if strategy_filter else None
    report = run_rebalance(asof=asof_date, dry_run=dry_run, strategies=strategies_arg)

    header = Table(title=f"Rebalance {report.asof} — {'DRY RUN' if dry_run else 'LIVE'}")
    header.add_column("Field")
    header.add_column("Value", justify="right")
    header.add_row("Account equity", f"${report.equity:,.2f}")
    header.add_row("Enabled strategies", str(len(report.enabled_strategies)))
    header.add_row("Total orders", str(report.total_orders))
    console.print(header)

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
        enabled_strategy_slugs,
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
        sample_slug = enabled_strategy_slugs()[:1]
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
            enabled_slugs=enabled_strategy_slugs(),
        )
        checks.append(("reconciliation", recon.ok, recon.detail))

    # 6. Risk limits.
    if settings_obj is not None:
        risk = check_risk_limits(
            data_dir=settings_obj.data_dir, enabled_slugs=enabled_strategy_slugs()
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


@cli.command(help="List all registered strategies.")
def strategies() -> None:
    table = Table(title="Registered strategies", show_header=True)
    table.add_column("Slug")
    table.add_column("Name")
    table.add_column("Rebalance")
    table.add_column("Universe size", justify="right")
    table.add_column("Live", justify="center")
    for spec in list_strategies():
        table.add_row(
            spec.slug,
            spec.name,
            spec.rebalance_frequency,
            str(len(spec.universe)),
            "yes" if spec.enabled_live else "no",
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    cli()
