# quant/intraday/cli.py
"""`quant intraday ...` command group."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console

from quant.intraday.data.quality import run_doctor
from quant.intraday.live.halt import clear_sleeve_halt, load_sleeve_halt, set_sleeve_halt
from quant.intraday.live.journal import read_ticks
from quant.util.config import Settings

console = Console()


def _data_dir() -> Path:
    return Settings().data_dir  # type: ignore[call-arg]


@click.group()
def intraday() -> None:
    """Intraday equities subsystem."""


@intraday.group()
def data() -> None:
    """Intraday data layer commands."""


@data.command()
def status() -> None:
    """Show intraday data store status."""
    root = _data_dir() / "intraday"
    counts = run_doctor(root)
    console.print(f"[bold]Intraday data[/bold] at {root}")
    for ds, n in counts.items():
        console.print(f"  {ds}: {n} partitions")


@data.command()
def doctor() -> None:
    """Health check: partition counts and obvious gaps."""
    root = _data_dir() / "intraday"
    counts = run_doctor(root)
    total = sum(counts.values())
    console.print(f"intraday store: {total} partitions across {len(counts)} datasets")
    for ds, n in counts.items():
        console.print(f"  {ds}: {n} partitions")


# ---------------------------------------------------------------------------
# live group
# ---------------------------------------------------------------------------


@intraday.group()
def live() -> None:
    """Intraday live-loop (sleeve) commands."""


@live.command("status")
def live_status() -> None:
    """Show sleeve halt state and the last journaled tick."""
    dd = _data_dir()
    halt = load_sleeve_halt(dd)
    click.echo(f"sleeve halt: {'HALTED — ' + halt.reason if halt.active else 'active (not halted)'}")
    df = read_ticks(dd)
    if df.empty:
        click.echo("no ticks journaled yet")
        return
    last = df.iloc[-1]
    click.echo(
        f"last tick {last['ts']}: day_pnl={last['day_pnl']:.2f} "
        f"round_trips={last['round_trips']} n_orders={last['n_orders']}"
    )


@live.command()
@click.option("--reason", required=True)
def halt(reason: str) -> None:
    """Halt the sleeve (stops the intraday loop only; daily system unaffected)."""
    set_sleeve_halt(_data_dir(), reason=reason, created_at=datetime.now(UTC))
    click.echo(f"sleeve halted: {reason}")


@live.command()
@click.option("--reason", required=True)
def resume(reason: str) -> None:
    """Clear the sleeve halt."""
    clear_sleeve_halt(_data_dir(), reason=reason)
    click.echo(f"sleeve resumed: {reason}")


@live.command()
def flat() -> None:
    """Manually flatten all sleeve positions (placeholder until wired to broker)."""
    click.echo("flatten requested — run via `live run` daemon path in production")


@live.command()
@click.option("--max-ticks", type=int, default=None, help="bound the run (default: forever)")
@click.option("--dry-run", is_flag=True, help="log orders without submitting")
def run(max_ticks: int | None, dry_run: bool) -> None:
    """Start the intraday tick loop. Wires real broker + feed + strategy."""
    from quant.execution.alpaca import AlpacaClient
    from quant.intraday.live.config import SleeveConfig
    from quant.intraday.live.feed import LiveQuoteFeed
    from quant.intraday.live.loop import TickDeps, recover_ledger, run_loop
    from quant.intraday.live.session import session_state
    from quant.intraday.live.strategy import MeanReversionStrategy

    cfg = SleeveConfig()
    dd = _data_dir()
    broker = AlpacaClient()
    feed = LiveQuoteFeed.from_settings(symbols=list(cfg.universe))
    strat = MeanReversionStrategy(cfg)
    ledger = recover_ledger(broker, cfg)

    def factory() -> TickDeps:
        now = datetime.now(UTC)
        ss = session_state(now)
        return TickDeps(
            data_dir=dd,
            config=cfg,
            broker=broker,
            feed=feed,
            strategy=strat,
            ledger=ledger,
            now=now,
            session_open=ss.open,
            session_close=ss.close,
            dry_run=dry_run,
        )

    run_loop(factory, max_ticks=max_ticks)
