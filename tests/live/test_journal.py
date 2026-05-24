"""Tests for the trade journal reader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.live.bookkeeping import append_trades
from quant.live.journal import read_journal


def _seed_trades(data: Path) -> None:
    append_trades(
        data,
        [
            {
                "date": pd.Timestamp(date(2026, 5, 20)),
                "strategy": "momentum",
                "symbol": "SPY",
                "side": "buy",
                "qty": 3,
                "client_order_id": "momentum-1",
                "dry_run": False,
            },
            {
                "date": pd.Timestamp(date(2026, 5, 24)),
                "strategy": "trend",
                "symbol": "TLT",
                "side": "sell",
                "qty": 1,
                "client_order_id": "trend-1",
                "dry_run": False,
            },
        ],
    )


def test_read_journal_filters_by_date(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_trades(data)
    df = read_journal(data, since=date(2026, 5, 23))
    assert len(df) == 1
    assert df.iloc[0]["strategy"] == "trend"


def test_read_journal_filters_by_strategy(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_trades(data)
    df = read_journal(data, strategy="momentum")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "SPY"


def test_read_journal_empty_when_no_file(tmp_path: Path) -> None:
    assert read_journal(tmp_path / "data").empty
