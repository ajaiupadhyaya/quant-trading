# quant/intraday/cli.py
"""`quant intraday ...` command group."""

from __future__ import annotations

import click
from rich.console import Console

from quant.intraday.data.quality import run_doctor
from quant.util.config import Settings

console = Console()


@click.group()
def intraday() -> None:
    """Intraday equities subsystem."""


@intraday.group()
def data() -> None:
    """Intraday data layer commands."""


@data.command()
def status() -> None:
    """Show intraday data store status."""
    root = Settings().data_dir / "intraday"  # type: ignore[call-arg]
    counts = run_doctor(root)
    console.print(f"[bold]Intraday data[/bold] at {root}")
    for ds, n in counts.items():
        console.print(f"  {ds}: {n} partitions")


@data.command()
def doctor() -> None:
    """Health check: partition counts and obvious gaps."""
    root = Settings().data_dir / "intraday"  # type: ignore[call-arg]
    counts = run_doctor(root)
    total = sum(counts.values())
    console.print(f"intraday store: {total} partitions across {len(counts)} datasets")
    for ds, n in counts.items():
        console.print(f"  {ds}: {n} partitions")
