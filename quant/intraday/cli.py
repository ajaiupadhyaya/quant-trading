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
def recon() -> None:
    """Summarize today's sleeve journal + reconcile ledger vs broker positions."""
    from quant.execution.alpaca import AlpacaClient
    from quant.intraday.live.config import SleeveConfig
    from quant.intraday.live.loop import recover_ledger
    from quant.intraday.live.recon import position_mismatches, summarize_day

    dd = _data_dir()
    s = summarize_day(dd)
    click.echo(f"ticks={s['n_ticks']} last_day_pnl={s['last_day_pnl']:.2f} "
               f"max_round_trips={s['max_round_trips']} halted_any={s['halted_any']}")
    cfg = SleeveConfig()
    broker = AlpacaClient()
    ledger = recover_ledger(broker, cfg)
    bad = position_mismatches(ledger.positions(), broker, cfg)
    click.echo("position recon: OK" if not bad else f"position MISMATCH: {bad}")
    click.echo("note: backtest-vs-live drift comparison deferred (needs intraday backtest baseline)")


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


# ---------------------------------------------------------------------------
# exec group
# ---------------------------------------------------------------------------


@intraday.group()
def exec_() -> None:
    """Optimal-execution (Almgren-Chriss) demonstration commands."""


intraday.add_command(exec_, name="exec")


def _demo_params(shares: int, horizon: int) -> tuple[object, float, float, float]:
    from quant.intraday.execution.calibrate import calibrate
    from quant.intraday.execution.config import ExecConfig

    cfg = ExecConfig(horizon_ticks=horizon)
    sigma, eta, gamma = calibrate(
        price=400.0,
        slice_shares=max(1, shares // horizon),
        adv_dollar=5_000_000_000.0,
        recent_returns=[0.0008, -0.0007, 0.0009, -0.0006] * 15,
        config=cfg,
    )
    return cfg, sigma, eta, gamma


@exec_.command()
@click.option("--symbol", required=True)
@click.option("--shares", type=int, required=True)
@click.option("--horizon", type=int, default=5)
@click.option("--lam", type=float, default=None, help="risk aversion (default ExecConfig)")
def schedule(symbol: str, shares: int, horizon: int, lam: float | None) -> None:
    """Print the Almgren-Chriss child-size schedule for a parent order."""
    from quant.intraday.execution.almgren_chriss import optimal_schedule
    from quant.intraday.execution.config import ExecConfig

    cfg, sigma, eta, gamma = _demo_params(shares, horizon)
    assert isinstance(cfg, ExecConfig)
    plan = optimal_schedule(
        total_shares=shares,
        n_intervals=horizon,
        tau=1.0,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        risk_aversion=lam if lam is not None else cfg.risk_aversion,
    )
    click.echo(f"A-C schedule for {symbol} ({shares} sh over {horizon} ticks):")
    for i, n in enumerate(plan.child_sizes):
        click.echo(f"  slice {i}: {n} sh")
    click.echo(f"expected_cost={plan.expected_cost:.4f} variance={plan.variance:.6f}")


@exec_.command()
@click.option("--symbol", required=True)
@click.option("--shares", type=int, required=True)
@click.option("--horizon", type=int, default=5)
def frontier(symbol: str, shares: int, horizon: int) -> None:
    """Print the efficient frontier (cost vs variance) + TWAP/immediate baselines."""
    from quant.intraday.execution.almgren_chriss import efficient_frontier
    from quant.intraday.execution.baselines import immediate, twap

    cfg, sigma, eta, gamma = _demo_params(shares, horizon)
    del cfg  # params extracted; ExecConfig not needed further
    pts = efficient_frontier(
        total_shares=shares,
        n_intervals=horizon,
        tau=1.0,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        lambdas=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    )
    click.echo(f"Efficient frontier for {symbol} ({shares} sh, horizon {horizon}):")
    click.echo("  lambda        expected_cost    variance")
    for p in pts:
        click.echo(f"  {p.risk_aversion:<12.1e} {p.expected_cost:<15.4f} {p.variance:.6f}")
    click.echo(f"baseline TWAP child sizes: {twap(total_shares=shares, n_intervals=horizon)}")
    click.echo(f"baseline immediate: {immediate(total_shares=shares)}")
    click.echo("(VWAP requires a volume curve; available in sim evaluation only.)")
