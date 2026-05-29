"""Tests for the append-only live/* parquet writers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.live.bookkeeping import (
    append_equity_row,
    append_trades,
    last_strategy_positions,
    read_equity,
    read_trades,
    write_strategy_positions,
)


def test_equity_append_round_trip(tmp_path: Path) -> None:
    data = tmp_path / "data"
    append_equity_row(
        data,
        asof=date(2026, 5, 24),
        equity=100_000.0,
        last_equity=99_500.0,
        cash=10_000.0,
        buying_power=200_000.0,
        portfolio_value=100_000.0,
    )
    append_equity_row(
        data,
        asof=date(2026, 5, 25),
        equity=100_500.0,
        last_equity=100_000.0,
        cash=9_500.0,
        buying_power=201_000.0,
        portfolio_value=100_500.0,
    )
    df = read_equity(data)
    assert len(df) == 2
    assert list(df["date"]) == sorted(df["date"])
    assert df.iloc[-1]["equity"] == 100_500.0


def test_trades_round_trip(tmp_path: Path) -> None:
    data = tmp_path / "data"
    rows = [
        {
            "date": pd.Timestamp(date(2026, 5, 24)),
            "strategy": "momentum",
            "symbol": "SPY",
            "side": "buy",
            "qty": 5,
            "client_order_id": "momentum-20260524-SPY-abcd",
            "dry_run": False,
        },
        {
            "date": pd.Timestamp(date(2026, 5, 24)),
            "strategy": "trend",
            "symbol": "TLT",
            "side": "sell",
            "qty": 2,
            "client_order_id": "trend-20260524-TLT-efgh",
            "dry_run": False,
        },
    ]
    append_trades(data, rows)
    df = read_trades(data)
    assert len(df) == 2
    assert set(df["strategy"]) == {"momentum", "trend"}


def test_append_trades_empty_is_noop(tmp_path: Path) -> None:
    append_trades(tmp_path / "data", [])
    assert not (tmp_path / "data" / "live" / "trades.parquet").exists()


def test_strategy_positions_snapshot_last_wins(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_strategy_positions(data, date(2026, 5, 24), "momentum", {"SPY": 10, "TLT": -5})
    write_strategy_positions(data, date(2026, 5, 25), "momentum", {"SPY": 12, "TLT": -3})
    write_strategy_positions(data, date(2026, 5, 25), "trend", {"GLD": 3})

    assert last_strategy_positions(data, "momentum") == {"SPY": 12, "TLT": -3}
    assert last_strategy_positions(data, "trend") == {"GLD": 3}
    assert last_strategy_positions(data, "unknown") == {}


def test_readers_on_empty_dir_return_empty_frames(tmp_path: Path) -> None:
    assert read_equity(tmp_path / "data").empty
    assert read_trades(tmp_path / "data").empty
    assert last_strategy_positions(tmp_path / "data", "any") == {}


def test_last_strategy_positions_returns_latest_same_day_write(tmp_data_dir: Path) -> None:
    d = date(2026, 5, 26)
    # Write 1 (manual run) holds GOOGL; an interleaved other-slug write separates the two mf writes.
    write_strategy_positions(tmp_data_dir, d, "multi-factor", {"BAC": 676, "GOOGL": 91})
    write_strategy_positions(tmp_data_dir, d, "trend", {"SPY": 70})
    # Write 2 (scheduled run) on the SAME date drops GOOGL, adds JNJ.
    write_strategy_positions(tmp_data_dir, d, "multi-factor", {"BAC": 670, "JNJ": 152})

    snap = last_strategy_positions(tmp_data_dir, "multi-factor")
    assert snap == {"BAC": 670, "JNJ": 152}  # latest write only; phantom GOOGL gone


def test_last_strategy_positions_single_write_unchanged(tmp_data_dir: Path) -> None:
    write_strategy_positions(tmp_data_dir, date(2026, 5, 26), "trend", {"SPY": 70, "DBC": 100})
    assert last_strategy_positions(tmp_data_dir, "trend") == {"SPY": 70, "DBC": 100}
