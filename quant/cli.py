"""Click CLI: top-level group + every subcommand wired to the strategy registry.

Foundation phase: most subcommands are stubs that raise `click.ClickException`
with a clear "not yet implemented in Plan N" message. `status` and `data` are
fully functional; the rest are scaffolded so the command surface is stable.
"""

from __future__ import annotations

import webbrowser
from datetime import date
from pathlib import Path

import click
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
    "--quick", is_flag=True, help="Skip combinatorial CV + bootstrap (Plan 3-only knobs)."
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
    if quick:
        # Plan 2 has no Plan-3 knobs to skip; flag is reserved for forward-compat.
        logger.info("--quick: skipping Plan-3 validation layers (none active in this plan).")

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    settings = Settings()  # type: ignore[call-arg]
    strategy_cls = REGISTRY[strategy]
    universe = list(strategy_cls.spec.universe)

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
        param_grid={},
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


@cli.command(help="Run the full validation battery (walk-forward + CPCV + DSR + ...).")
@click.argument("strategy")
def validate(strategy: str) -> None:
    _require_strategy(strategy)
    raise click.ClickException(
        f"validate is not implemented in Foundation. "
        f"Plan 3 (validation) will fill this in. (strategy={strategy})"
    )


@cli.command(help="Run today's live rebalance across all enabled strategies.")
@click.option("--dry-run", is_flag=True, help="Print orders only; do not submit.")
def rebalance(dry_run: bool) -> None:
    raise click.ClickException(
        "rebalance is not implemented in Foundation. Plan 6 will wire it up."
    )


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


@cli.command(help="Print the structured trade journal.")
@click.option("--since", default=None, help="Filter trades since YYYY-MM-DD.")
def journal(since: str | None) -> None:
    raise click.ClickException(
        "journal is not implemented in Foundation. Plan 6 will fill this in."
    )


@cli.command(help="Open the Textual TUI monitor.")
def monitor() -> None:
    raise click.ClickException(
        "monitor is not implemented in Foundation. Plan 6 will fill this in."
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
