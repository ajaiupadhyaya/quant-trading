"""Textual TUI: the ``quant monitor`` command.

A multi-pane terminal dashboard. Refreshes every ``REFRESH_SECONDS`` from
Alpaca + the local ``data/live/*.parquet`` bookkeeping; ``r`` forces an
immediate refresh, ``q`` quits.

The data layer is intentionally factored into ``MonitorSnapshot.build`` so
tests can construct a snapshot without ever running the Textual event loop.
"""

from __future__ import annotations

import json
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
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static

from quant.execution.alpaca import AccountInfo, AlpacaClient, PositionRow
from quant.governance.models import GovernanceError
from quant.governance.store import (
    allocation_path,
    drift_report_path,
    load_allocation,
    load_strategy_states,
    strategy_states_path,
)
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
    governance_state: str = "unknown"
    allocation: float = 0.0
    drift_flag: str = "unknown"


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
        try:
            states = load_strategy_states(strategy_states_path(data_dir))
        except GovernanceError:
            states = {}
        try:
            allocation = load_allocation(allocation_path(data_dir))
        except GovernanceError:
            allocation = {}
        drift_flags = _load_drift_flags(data_dir)
        strategies = [
            StrategySnapshot(
                slug=spec.slug,
                name=spec.name,
                enabled_live=spec.enabled_live,
                n_positions=int(per_strategy_counts.get(spec.slug, 0)),
                governance_state=states[spec.slug].state.value
                if spec.slug in states
                else "unknown",
                allocation=float(allocation.get(spec.slug, 0.0)),
                drift_flag=drift_flags.get(spec.slug, "unknown"),
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


def render_strategies_table(
    strategies: list[StrategySnapshot],
    selected: str | None = None,
) -> Table:
    table = Table(title="Strategies", expand=True)
    table.add_column("#", justify="right")
    table.add_column("Slug")
    table.add_column("Name")
    table.add_column("Live", justify="center")
    table.add_column("Gov", justify="center")
    table.add_column("Alloc", justify="right")
    table.add_column("Drift", justify="center")
    table.add_column("# Pos", justify="right")
    for i, s in enumerate(strategies, start=1):
        live = "[green]on[/]" if s.enabled_live else "[dim]off[/]"
        gov_style = "green" if s.governance_state == "live" else "yellow"
        slug_display = f"[reverse]{s.slug}[/]" if s.slug == selected else s.slug
        table.add_row(
            str(i),
            slug_display,
            s.name,
            live,
            f"[{gov_style}]{s.governance_state}[/]",
            f"{s.allocation * 100:.1f}%",
            s.drift_flag,
            str(s.n_positions),
        )
    return table


def _load_drift_flags(data_dir: Path) -> dict[str, str]:
    path = drift_report_path(data_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    priority = {"halt_candidate": 3, "watch": 2, "normal": 1}
    out: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        slug = raw.get("strategy")
        flag = raw.get("flag")
        if not isinstance(slug, str) or not isinstance(flag, str):
            continue
        current = out.get(slug, "normal")
        if priority.get(flag, 0) >= priority.get(current, 0):
            out[slug] = flag
    return out


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


def render_trades_table(trades: pd.DataFrame, title_suffix: str = "") -> Table:
    table = Table(title=f"Trades today ({len(trades)}){title_suffix}", expand=True)
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
        Binding("b", "open_tearsheet", "Open tearsheet"),
        Binding("question_mark", "help", "Help"),
        Binding("0", "clear_filter", "All strategies"),
        Binding("1", "select_index('1')", "Strat 1", show=False),
        Binding("2", "select_index('2')", "Strat 2", show=False),
        Binding("3", "select_index('3')", "Strat 3", show=False),
        Binding("4", "select_index('4')", "Strat 4", show=False),
        Binding("5", "select_index('5')", "Strat 5", show=False),
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
        # Currently-selected strategy slug for drill-down. None means "all".
        self._strategy_filter: str | None = None
        # Cache the last snapshot so tear-sheet + filter actions don't re-fetch.
        self._last_snapshot: MonitorSnapshot | None = None

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

    def action_help(self) -> None:
        self.push_screen(HelpOverlay())

    def action_clear_filter(self) -> None:
        self._strategy_filter = None
        self._render(self._last_snapshot) if self._last_snapshot else self._refresh()

    def action_select_index(self, idx: str) -> None:
        if self._last_snapshot is None:
            return
        slugs = [s.slug for s in self._last_snapshot.strategies]
        i = int(idx) - 1
        if 0 <= i < len(slugs):
            self._strategy_filter = slugs[i]
            self._render(self._last_snapshot)

    def action_open_tearsheet(self) -> None:
        """Open the current strategy's tear-sheet (or the combined book if no filter)."""
        import webbrowser

        slug = self._strategy_filter or "_combined"
        path = self._data_dir / "backtests" / slug / "tearsheet.html"
        if path.exists():
            webbrowser.open(path.resolve().as_uri())
            self.notify(f"Opened {slug} tear-sheet")
        else:
            self.notify(f"No tear-sheet at {path}", severity="warning")

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
        self._last_snapshot = snap
        self._render(snap)

    def _render(self, snap: MonitorSnapshot | None) -> None:
        if snap is None:
            return
        today_pnl = snap.account.equity - snap.account.last_equity

        # Apply the optional drill-down filter on positions + trades.
        slug_filter = self._strategy_filter
        filtered_trades = snap.today_trades
        if slug_filter is not None and not snap.today_trades.empty:
            filtered_trades = snap.today_trades[snap.today_trades["strategy"] == slug_filter]

        if self._account_pane is not None:
            self._account_pane.update(render_account_table(snap.account, today_pnl))
        if self._strategies_pane is not None:
            self._strategies_pane.update(
                render_strategies_table(snap.strategies, selected=slug_filter)
            )
        if self._positions_pane is not None:
            self._positions_pane.update(render_positions_table(snap.positions))
        if self._trades_pane is not None:
            title_suffix = f" [{slug_filter}]" if slug_filter else ""
            self._trades_pane.update(
                render_trades_table(filtered_trades, title_suffix=title_suffix)
            )
        if self._equity_pane is not None:
            self._equity_pane.update(render_equity_sparkline(snap.equity_history))


class HelpOverlay(ModalScreen[None]):
    """Modal help screen — dismissed by any key."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "\n".join(
                [
                    "[bold]Quant Monitor — keybindings[/]",
                    "",
                    "  [cyan]r[/]      refresh now (also auto-refreshes every 60s)",
                    "  [cyan]b[/]      open the current selection's tear-sheet in your browser",
                    "  [cyan]1-5[/]    drill down into the Nth registered strategy",
                    "  [cyan]0[/]      clear drill-down filter (show all strategies)",
                    "  [cyan]?[/]      this help",
                    "  [cyan]q[/]      quit",
                    "",
                    "[dim]press any of escape / q / ? to dismiss[/]",
                ]
            ),
        )

    def action_dismiss(self) -> None:  # type: ignore[override]
        self.app.pop_screen()
