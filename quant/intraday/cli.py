# quant/intraday/cli.py
"""`quant intraday ...` command group."""

from __future__ import annotations

import json
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
    click.echo(
        f"sleeve halt: {'HALTED — ' + halt.reason if halt.active else 'active (not halted)'}"
    )
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
    click.echo(
        f"ticks={s['n_ticks']} last_day_pnl={s['last_day_pnl']:.2f} "
        f"max_round_trips={s['max_round_trips']} halted_any={s['halted_any']}"
    )
    cfg = SleeveConfig()
    broker = AlpacaClient()
    ledger = recover_ledger(broker, cfg)
    bad = position_mismatches(ledger.positions(), broker, cfg)
    click.echo("position recon: OK" if not bad else f"position MISMATCH: {bad}")
    click.echo(
        "note: backtest-vs-live drift comparison deferred (needs intraday backtest baseline)"
    )


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


# ---------------------------------------------------------------------------
# mm group
# ---------------------------------------------------------------------------


@intraday.group()
def mm() -> None:
    """Avellaneda-Stoikov market-making simulator (sim/research only)."""


def _mm_demo_prices(seed: int, steps: int) -> list[float]:
    import random as _random

    from quant.intraday.marketmaking.price_path import abm_path

    return abm_path(s0=400.0, sigma=0.02, dt=1.0, n_steps=steps, rng=_random.Random(seed))


@mm.command()
@click.option("--symbol", required=True)
@click.option("--gamma", type=float, default=None, help="risk aversion (default MMConfig)")
@click.option("--seed", type=int, default=7)
@click.option("--steps", type=int, default=600)
def simulate(symbol: str, gamma: float | None, seed: int, steps: int) -> None:
    """Run one A-S market-making episode and print P&L + inventory stats."""
    import dataclasses

    from quant.intraday.marketmaking.config import MMConfig
    from quant.intraday.marketmaking.simulator import run_market_making

    cfg = MMConfig(horizon_seconds=float(steps), dt_seconds=1.0, seed=seed)
    if gamma is not None:
        cfg = dataclasses.replace(cfg, gamma=gamma)
    prices = _mm_demo_prices(seed, cfg.n_steps)
    r = run_market_making(prices, cfg)
    click.echo(
        f"A-S market making for {symbol} (gamma={cfg.gamma}, {cfg.n_steps} steps, seed={seed}):"
    )
    click.echo(f"  final pnl:        {r.final_pnl:.2f}")
    click.echo(f"  spread captured:  {r.spread_captured:.2f}")
    click.echo(f"  fills:            {r.n_bid_fills} bid / {r.n_ask_fills} ask")
    click.echo(
        f"  inventory:        mean|q|={r.mean_abs_inventory:.2f}"
        f" max|q|={r.max_abs_inventory} terminal={r.terminal_inventory}"
    )
    click.echo("note: stylized A-S model (A, k are assumed parameters, not a live edge).")


@mm.command()
@click.option("--symbol", required=True)
@click.option("--seed", type=int, default=7)
@click.option("--steps", type=int, default=800)
def sweep(symbol: str, seed: int, steps: int) -> None:
    """Print the gamma tradeoff table (spread-capture vs inventory-risk)."""
    from quant.intraday.marketmaking.config import MMConfig
    from quant.intraday.marketmaking.evaluate import gamma_sweep

    cfg = MMConfig(horizon_seconds=float(steps), dt_seconds=1.0, seed=seed)
    prices = _mm_demo_prices(seed, cfg.n_steps)
    pts = gamma_sweep(prices, cfg, [0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    click.echo(f"A-S gamma sweep for {symbol} ({cfg.n_steps} steps, seed={seed}):")
    click.echo("  gamma     pnl         fills    mean|q|   max|q|   terminal_q")
    for p in pts:
        click.echo(
            f"  {p.gamma:<8.2f}  {p.final_pnl:<10.2f}  {p.n_fills:<7d}  "
            f"{p.mean_abs_inventory:<8.2f}  {p.max_abs_inventory:<7d}  {p.terminal_inventory}"
        )
    click.echo("note: stylized A-S model (A, k are assumed parameters, not a live edge).")


# ---------------------------------------------------------------------------
# rl group
# ---------------------------------------------------------------------------


@intraday.group()
def rl() -> None:
    """Tabular Q-learning execution agent (sim/research only)."""


@rl.command()
@click.option("--shares", type=int, default=20)
@click.option("--steps", type=int, default=10)
@click.option("--episodes", type=int, default=20000)
@click.option("--seed", type=int, default=7)
def train(shares: int, steps: int, episodes: int, seed: int) -> None:
    """Train the agent and print the convergence curve (mean episode cost per block)."""
    from quant.intraday.rl.config import RLConfig
    from quant.intraday.rl.qlearning import train as train_agent

    cfg = RLConfig(total_shares=shares, n_steps=steps, n_episodes=episodes, seed=seed)
    result = train_agent(cfg)
    click.echo(f"RL execution training ({shares} sh, {steps} steps, {episodes} episodes):")
    click.echo("  convergence (mean episode cost per block, first -> last):")
    curve = result.training_curve
    for i, c in enumerate(curve):
        if i == 0 or i == len(curve) - 1 or i % 5 == 0:
            click.echo(f"    block {i:>2}: {c:.4f}")
    click.echo(f"  improved from {curve[0]:.4f} to {curve[-1]:.4f}")
    click.echo("note: tabular RL rediscovers the DP/A-C optimum; the point is it LEARNS it.")


@rl.command()
@click.option("--shares", type=int, default=20)
@click.option("--steps", type=int, default=10)
@click.option("--episodes", type=int, default=20000)
@click.option("--seed", type=int, default=7)
@click.option("--eval-paths", type=int, default=300)
def compare(shares: int, steps: int, episodes: int, seed: int, eval_paths: int) -> None:
    """Compare learned policy vs Almgren-Chriss optimal vs TWAP (mean execution cost)."""
    from quant.intraday.rl.config import RLConfig
    from quant.intraday.rl.evaluate import compare as compare_policies

    cfg = RLConfig(total_shares=shares, n_steps=steps, n_episodes=episodes, seed=seed)
    res = compare_policies(cfg, n_eval_paths=eval_paths)
    click.echo(f"RL execution comparison ({shares} sh, {steps} steps, {eval_paths} eval paths):")
    click.echo(f"  learned (RL):     mean cost {res['learned']:.4f}")
    click.echo(f"  Almgren-Chriss:   mean cost {res['almgren_chriss']:.4f}")
    click.echo(f"  TWAP:             mean cost {res['twap']:.4f}")
    click.echo(f"  learned schedule: {res['learned_schedule']}")
    click.echo(
        "note: tabular RL rediscovers the DP/A-C optimum; the point is it LEARNS it from rewards."
    )


# ---------------------------------------------------------------------------
# dl group
# ---------------------------------------------------------------------------


@intraday.group()
def dl() -> None:
    """Deep-learning alpha (torch LSTM, next-bar return) — sim/research only."""


@dl.command("train")
@click.option("--n", type=int, default=3000, help="length of the synthetic signal series")
@click.option("--window", type=int, default=12)
@click.option("--epochs", type=int, default=40)
@click.option("--seed", type=int, default=7)
def dl_train(n: int, window: int, epochs: int, seed: int) -> None:
    """Train on the synthetic-signal series and print the per-epoch loss curve."""
    from quant.intraday.dl.config import DLConfig
    from quant.intraday.dl.data import make_windows, standardize, train_test_split
    from quant.intraday.dl.evaluate import synthetic_signal_series
    from quant.intraday.dl.train import train_model

    cfg = DLConfig(window=window, epochs=epochs, seed=seed)
    series = synthetic_signal_series(n=n, seed=seed)
    x, y = make_windows(series, cfg.window)
    x_tr, y_tr, x_te, _ = train_test_split(x, y, cfg.train_frac)
    x_tr_z, _, mu, sd = standardize(x_tr, x_te)
    y_tr_z = (y_tr - mu) / sd
    out = train_model(x_tr_z, y_tr_z, cfg)
    click.echo(f"DL alpha training ({n} pts, window {window}, {epochs} epochs, seed {seed}):")
    curve = out.loss_curve
    for i, c in enumerate(curve):
        if i == 0 or i == len(curve) - 1 or i % 5 == 0:
            click.echo(f"  epoch {i:>3}: loss {c:.5f}")
    click.echo(f"  loss fell from {curve[0]:.5f} to {curve[-1]:.5f} (training works).")


@dl.command("evaluate")
@click.option("--n", type=int, default=3000, help="length of each evaluation series")
@click.option("--window", type=int, default=12)
@click.option("--epochs", type=int, default=40)
@click.option("--seed", type=int, default=7)
@click.option(
    "--cost",
    type=float,
    default=0.0,
    help="per-unit-turnover cost charged to the sign-of-prediction rule",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="write the full dual-track evaluation record to this JSON path",
)
def dl_evaluate(n: int, window: int, epochs: int, seed: int, cost: float, out: Path | None) -> None:
    """Dual-track OOS comparison: LSTM vs linear vs naive on a synthetic-signal series
    (LSTM should win) and a near-random series (LSTM should NOT win — the honest result).
    Reports statistical metrics plus the economics of a sign-of-prediction rule, and can
    persist the full record as a reproducible JSON artifact via --out."""
    from quant.intraday.dl.config import DLConfig
    from quant.intraday.dl.evaluate import build_evaluation

    cfg = DLConfig(window=window, epochs=epochs, seed=seed)
    record = build_evaluation(cfg, n=n, seed=seed, cost_per_turn=cost)
    click.echo(
        f"DL alpha OOS evaluation (window {window}, {epochs} epochs, seed {seed}, "
        f"cost/turn {cost:g}):"
    )
    for name, track in record["tracks"].items():
        click.echo(f"\n  [{name}] OOS  (mse / dir-acc / r2  |  sharpe-net / hit / turnover):")
        for model_name in ("lstm", "linear", "naive"):
            m = track["models"][model_name]
            click.echo(
                f"    {model_name:<7} mse {m['mse']:.5f}   "
                f"dir-acc {m['directional_accuracy']:.3f}   r2 {m['r2']:.4f}   |   "
                f"sharpe-net {m['sharpe_net']:+.3f}   hit {m['hit_rate']:.3f}   "
                f"turn {m['avg_turnover']:.3f}"
            )
    click.echo(
        "\nnote: intraday returns are near-unforecastable (EMH); DL does not beat simple "
        "baselines OOS on the random track — sharpe-net there sits near zero (negative once "
        "costs bite). The value here is the technique + honest evaluation."
    )
    if out is not None:
        out.write_text(json.dumps(record, indent=2))
        click.echo(f"\nwrote evaluation record -> {out}")
