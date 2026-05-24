"""Textual TUI: the ``quant monitor`` command.

A multi-pane terminal dashboard. Refreshes every ``REFRESH_SECONDS`` from
Alpaca + the local ``data/live/*.parquet`` bookkeeping; ``r`` forces an
immediate refresh, ``q`` quits.

The data layer is intentionally factored into ``MonitorSnapshot.build`` so
tests can construct a snapshot without ever running the Textual event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from quant.execution.alpaca import AccountInfo, AlpacaClient, PositionRow
from quant.live.bookkeeping import read_equity, read_trades
from quant.strategies import list_strategies
from quant.util.config import Settings

REFRESH_SECONDS = 60


@dataclass(frozen=True)
class StrategySnapshot:
    slug: str
    name: str
    enabled_live: bool
    n_positions: int


@dataclass
class MonitorSnapshot:
    """All the data needed to render one frame of the TUI."""

    asof: date
    account: AccountInfo
    positions: list[PositionRow]
    strategies: list[StrategySnapshot]
    today_trades: pd.DataFrame
    equity_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def build(
        cls,
        *,
        client: AlpacaClient,
        data_dir: Path,
        asof: date | None = None,
    ) -> MonitorSnapshot:
        asof = asof or date.today()
        account = client.account()
        positions = client.positions()

        equity_hist = read_equity(data_dir)
        trades = read_trades(data_dir)

        if not trades.empty:
            mask = trades["date"] >= pd.Timestamp(asof)
            today_trades = trades[mask].reset_index(drop=True)
        else:
            today_trades = trades

        per_strategy_counts: dict[str, int] = {}
        if not trades.empty:
            raw = trades.groupby("strategy")["symbol"].nunique().to_dict()
            per_strategy_counts = {str(k): int(v) for k, v in raw.items()}
        strategies = [
            StrategySnapshot(
                slug=spec.slug,
                name=spec.name,
                enabled_live=spec.enabled_live,
                n_positions=int(per_strategy_counts.get(spec.slug, 0)),
            )
            for spec in list_strategies()
        ]

        return cls(
            asof=asof,
            account=account,
            positions=positions,
            strategies=strategies,
            today_trades=today_trades,
            equity_history=equity_hist,
        )


# ---- pure-function renderers (tested independently of Textual) ---------------


def render_account_table(account: AccountInfo, today_pnl: float) -> Table:
    table = Table(title="Account", show_header=False, expand=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Equity", f"${account.equity:,.2f}")
    pnl_color = "green" if today_pnl >= 0 else "red"
    table.add_row("Today P&L", f"[{pnl_color}]${today_pnl:+,.2f}[/]")
    table.add_row("Cash", f"${account.cash:,.2f}")
    table.add_row("Buying Power", f"${account.buying_power:,.2f}")
    table.add_row("Pattern Day Trader", "yes" if account.pattern_day_trader else "no")
    return table


def render_strategies_table(strategies: list[StrategySnapshot]) -> Table:
    table = Table(title="Strategies", expand=True)
    table.add_column("Slug")
    table.add_column("Name")
    table.add_column("Live", justify="center")
    table.add_column("# Pos", justify="right")
    for s in strategies:
        live = "[green]on[/]" if s.enabled_live else "[dim]off[/]"
        table.add_row(s.slug, s.name, live, str(s.n_positions))
    return table


def render_positions_table(positions: list[PositionRow]) -> Table:
    table = Table(title=f"Positions ({len(positions)})", expand=True)
    for col in ("Symbol", "Qty", "Avg", "Last", "Mkt Value", "Unrealized P&L"):
        table.add_column(col, justify="right" if col != "Symbol" else "left")
    for p in positions:
        color = "green" if p.unrealized_pl >= 0 else "red"
        table.add_row(
            p.symbol,
            str(p.qty),
            f"${p.avg_entry_price:,.2f}",
            f"${p.current_price:,.2f}",
            f"${p.market_value:,.2f}",
            f"[{color}]${p.unrealized_pl:+,.2f}[/]",
        )
    return table


def render_trades_table(trades: pd.DataFrame) -> Table:
    table = Table(title=f"Trades today ({len(trades)})", expand=True)
    for col in ("Time", "Strategy", "Symbol", "Side", "Qty"):
        table.add_column(col)
    if trades.empty:
        table.add_row("—", "—", "—", "—", "—")
        return table
    for row in trades.itertuples(index=False):
        ts = pd.Timestamp(str(row.date))
        table.add_row(
            ts.date().isoformat(),
            str(row.strategy),
            str(row.symbol),
            str(row.side),
            str(row.qty),
        )
    return table


def render_equity_sparkline(history: pd.DataFrame, width: int = 60) -> Text:
    """Plain-ASCII sparkline of the equity curve."""
    if history.empty or "equity" not in history.columns:
        return Text("(no equity history)", style="dim")
    values = history["equity"].astype(float).tail(width).tolist()
    if len(values) < 2:
        return Text(f"equity={values[-1]:,.2f}" if values else "(empty)", style="dim")
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-9)
    levels = "▁▂▃▄▅▆▇█"
    chars = []
    for v in values:
        idx = int((v - lo) / span * (len(levels) - 1))
        chars.append(levels[idx])
    label = f"  equity {values[-1]:,.2f}  ({len(values)} pts, lo {lo:,.2f}  hi {hi:,.2f})"
    text = Text("".join(chars))
    text.append(label, style="dim")
    return text


# ---- Textual app -----------------------------------------------------------


class QuantMonitor(App[None]):
    """Multi-pane Textual monitor for the live paper-trading state."""

    CSS = """
    Screen { layout: vertical; }
    #top { height: 40%; }
    #middle { height: 35%; }
    #bottom { height: 25%; }
    .pane { border: round $accent; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        *,
        client: AlpacaClient | None = None,
        data_dir: Path | None = None,
        asof: date | None = None,
    ) -> None:
        super().__init__()
        settings = Settings()  # type: ignore[call-arg]
        self._client = client or AlpacaClient(settings=settings)
        self._data_dir = data_dir or settings.data_dir
        self._asof = asof
        self._account_pane: Static | None = None
        self._strategies_pane: Static | None = None
        self._positions_pane: Static | None = None
        self._trades_pane: Static | None = None
        self._equity_pane: Static | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="top"):
                self._account_pane = Static(classes="pane", id="account")
                self._strategies_pane = Static(classes="pane", id="strategies")
                yield self._account_pane
                yield self._strategies_pane
            with Horizontal(id="middle"):
                self._positions_pane = Static(classes="pane", id="positions")
                self._trades_pane = Static(classes="pane", id="trades")
                yield self._positions_pane
                yield self._trades_pane
            with Horizontal(id="bottom"):
                self._equity_pane = Static(classes="pane", id="equity")
                yield self._equity_pane
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(REFRESH_SECONDS, self._refresh)
        self._refresh()

    def action_refresh(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        try:
            snap = MonitorSnapshot.build(
                client=self._client,
                data_dir=self._data_dir,
                asof=self._asof,
            )
        except Exception as exc:  # network or auth flake — keep app alive
            for pane in (
                self._account_pane,
                self._strategies_pane,
                self._positions_pane,
                self._trades_pane,
                self._equity_pane,
            ):
                if pane is not None:
                    pane.update(f"[red]refresh failed: {exc!r}[/]")
            return

        today_pnl = snap.account.equity - snap.account.last_equity
        if self._account_pane is not None:
            self._account_pane.update(render_account_table(snap.account, today_pnl))
        if self._strategies_pane is not None:
            self._strategies_pane.update(render_strategies_table(snap.strategies))
        if self._positions_pane is not None:
            self._positions_pane.update(render_positions_table(snap.positions))
        if self._trades_pane is not None:
            self._trades_pane.update(render_trades_table(snap.today_trades))
        if self._equity_pane is not None:
            self._equity_pane.update(render_equity_sparkline(snap.equity_history))
