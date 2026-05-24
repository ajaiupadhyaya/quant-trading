"""Tests for the TUI's pure-function renderers and the snapshot builder.

The Textual ``QuantMonitor`` event loop is not exercised here — these tests
cover the data shapes the panes render so that a future Textual-API change
can be spotted without booting the app.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from rich.table import Table
from rich.text import Text

from quant.execution.alpaca import AccountInfo, PositionRow
from quant.live.bookkeeping import append_equity_row, append_trades
from quant.tui import (
    MonitorSnapshot,
    render_account_table,
    render_equity_sparkline,
    render_positions_table,
    render_strategies_table,
    render_trades_table,
)


class _StubClient:
    def __init__(
        self,
        account: AccountInfo,
        positions: list[PositionRow] | None = None,
    ) -> None:
        self._account = account
        self._positions = positions or []

    def account(self) -> AccountInfo:
        return self._account

    def positions(self) -> list[PositionRow]:
        return list(self._positions)


def _acct(equity: float = 100_000.0, last: float = 99_500.0) -> AccountInfo:
    return AccountInfo(
        equity=equity,
        last_equity=last,
        buying_power=equity * 2,
        cash=equity * 0.1,
        portfolio_value=equity,
        pattern_day_trader=False,
    )


def test_render_account_table_returns_rich_table() -> None:
    table = render_account_table(_acct(), today_pnl=500.0)
    assert isinstance(table, Table)
    # Title and column count assertions are stable across rich versions.
    assert table.title == "Account"
    assert len(table.columns) == 2


def test_render_positions_table_with_no_positions() -> None:
    table = render_positions_table([])
    assert "Positions (0)" in str(table.title)


def test_render_trades_table_handles_empty_frame() -> None:
    table = render_trades_table(pd.DataFrame(columns=["date", "strategy", "symbol", "side", "qty"]))
    assert "Trades today (0)" in str(table.title)


def test_render_strategies_table_includes_all_specs() -> None:
    from quant.strategies import list_strategies
    from quant.tui import StrategySnapshot

    snaps = [
        StrategySnapshot(slug=s.slug, name=s.name, enabled_live=s.enabled_live, n_positions=0)
        for s in list_strategies()
    ]
    table = render_strategies_table(snaps)
    assert isinstance(table, Table)


def test_equity_sparkline_renders_for_short_history() -> None:
    history = pd.DataFrame({"equity": [100.0, 101.0, 99.0, 105.0]})
    text = render_equity_sparkline(history, width=10)
    assert isinstance(text, Text)
    assert "equity" in str(text)


def test_equity_sparkline_empty_returns_text() -> None:
    text = render_equity_sparkline(pd.DataFrame())
    assert isinstance(text, Text)
    assert "no equity history" in str(text)


def test_monitor_snapshot_builds_from_stubs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    append_equity_row(
        data,
        asof=date(2026, 5, 24),
        equity=100_000.0,
        last_equity=99_000.0,
        cash=10_000.0,
        buying_power=200_000.0,
        portfolio_value=100_000.0,
    )
    append_trades(
        data,
        [
            {
                "date": pd.Timestamp(date(2026, 5, 24)),
                "strategy": "momentum",
                "symbol": "SPY",
                "side": "buy",
                "qty": 5,
                "client_order_id": "momentum-coid-1",
                "dry_run": False,
            }
        ],
    )

    client = _StubClient(
        account=_acct(equity=100_500.0, last=100_000.0),
        positions=[
            PositionRow(
                symbol="SPY",
                qty=5,
                avg_entry_price=500.0,
                market_value=2_550.0,
                unrealized_pl=50.0,
                current_price=510.0,
                side="long",
            )
        ],
    )

    snap = MonitorSnapshot.build(client=client, data_dir=data, asof=date(2026, 5, 24))  # type: ignore[arg-type]
    assert snap.account.equity == 100_500.0
    assert len(snap.positions) == 1
    assert len(snap.today_trades) == 1
    # Strategies snapshot reflects the registry + per-strategy counts.
    momentum = next(s for s in snap.strategies if s.slug == "momentum")
    assert momentum.n_positions == 1
