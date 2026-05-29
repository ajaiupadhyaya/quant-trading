"""Tests for the pre-trade safety checks."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.execution.alpaca import PositionRow
from quant.live.bookkeeping import append_equity_row, write_strategy_positions
from quant.live.safety import (
    StrategyRiskBudget,
    check_bar_freshness,
    check_market_open,
    check_reconciliation,
    check_risk_limits,
)


def test_check_market_open_skips_weekends() -> None:
    res = check_market_open(date(2024, 1, 6))  # Saturday
    assert not res.ok
    assert "not a NYSE trading day" in res.detail


def test_check_market_open_passes_weekday() -> None:
    res = check_market_open(date(2024, 6, 28))  # Friday
    assert res.ok


def test_reconciliation_first_run_passes(tmp_path: Path) -> None:
    res = check_reconciliation(
        data_dir=tmp_path / "data",
        alpaca_positions=[],
        enabled_slugs=["momentum"],
    )
    assert res.ok
    assert "no prior snapshots" in res.detail


def test_reconciliation_matches_within_tolerance(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_strategy_positions(data, date(2026, 5, 25), "momentum", {"SPY": 10, "TLT": -5})
    write_strategy_positions(data, date(2026, 5, 25), "trend", {"GLD": 4})

    alpaca = [
        PositionRow(
            symbol="SPY",
            qty=10,
            avg_entry_price=500.0,
            market_value=5000.0,
            unrealized_pl=0,
            current_price=500.0,
            side="long",
        ),
        PositionRow(
            symbol="TLT",
            qty=-5,
            avg_entry_price=90.0,
            market_value=-450.0,
            unrealized_pl=0,
            current_price=90.0,
            side="short",
        ),
        PositionRow(
            symbol="GLD",
            qty=4,
            avg_entry_price=180.0,
            market_value=720.0,
            unrealized_pl=0,
            current_price=180.0,
            side="long",
        ),
    ]
    res = check_reconciliation(
        data_dir=data, alpaca_positions=alpaca, enabled_slugs=["momentum", "trend"]
    )
    assert res.ok


def test_reconciliation_flags_mismatch(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_strategy_positions(data, date(2026, 5, 25), "momentum", {"SPY": 100})
    alpaca = [
        PositionRow(
            symbol="SPY",
            qty=80,  # 20-share mismatch — clearly outside tolerance
            avg_entry_price=500.0,
            market_value=40000.0,
            unrealized_pl=0,
            current_price=500.0,
            side="long",
        ),
    ]
    res = check_reconciliation(data_dir=data, alpaca_positions=alpaca, enabled_slugs=["momentum"])
    assert not res.ok
    assert "SPY" in res.detail


def test_risk_limits_pass_within_budget(tmp_path: Path) -> None:
    data = tmp_path / "data"
    # Equity history with mild drawdown (-5%)
    for i, eq in enumerate([100_000, 99_000, 95_000, 96_000]):
        append_equity_row(
            data,
            asof=date(2026, 5, 20 + i),
            equity=eq,
            last_equity=eq,
            cash=eq * 0.1,
            buying_power=eq * 2,
            portfolio_value=eq,
        )
    res = check_risk_limits(
        data_dir=data,
        enabled_slugs=["momentum", "trend"],
        budget=StrategyRiskBudget(max_drawdown=0.25),
    )
    assert res.ok
    assert not res.halted_strategies


def test_risk_limits_trip_on_deep_drawdown(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for i, eq in enumerate([100_000, 90_000, 80_000, 70_000]):
        append_equity_row(
            data,
            asof=date(2026, 5, 20 + i),
            equity=eq,
            last_equity=eq,
            cash=eq * 0.1,
            buying_power=eq * 2,
            portfolio_value=eq,
        )
    res = check_risk_limits(
        data_dir=data,
        enabled_slugs=["momentum", "trend"],
        budget=StrategyRiskBudget(max_drawdown=0.25),  # -30% dd > -25%
    )
    assert not res.ok
    assert res.halted_strategies == frozenset({"momentum", "trend"})


def test_bar_freshness_missing_dir(tmp_path: Path) -> None:
    res = check_bar_freshness(tmp_path / "data", symbols=["SPY"], asof=date(2026, 5, 25))
    assert not res.ok
    assert "does not exist" in res.detail


def test_bar_freshness_recent_pass(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-24")], name="timestamp"),
    )
    df.to_parquet(raw / "SPY.parquet")
    res = check_bar_freshness(
        tmp_path / "data", symbols=["SPY"], asof=date(2026, 5, 25), max_age_days=4
    )
    assert res.ok


def test_reconciliation_counts_winddown_orphan_as_expected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_strategy_positions(data, date(2026, 5, 26), "defensive-etf-allocation", {"SPY": 10})
    write_strategy_positions(data, date(2026, 5, 26), "trend", {"DBC": 1000})
    alpaca = [
        PositionRow(
            symbol="SPY",
            qty=10,
            avg_entry_price=500.0,
            market_value=5000.0,
            unrealized_pl=0.0,
            current_price=500.0,
            side="long",
        ),
        PositionRow(
            symbol="DBC",
            qty=1000,
            avg_entry_price=20.0,
            market_value=20000.0,
            unrealized_pl=0.0,
            current_price=20.0,
            side="long",
        ),
    ]
    bad = check_reconciliation(
        data_dir=data, alpaca_positions=alpaca, enabled_slugs=["defensive-etf-allocation"]
    )
    assert not bad.ok  # DBC unexpected without winddown_slugs
    good = check_reconciliation(
        data_dir=data,
        alpaca_positions=alpaca,
        enabled_slugs=["defensive-etf-allocation"],
        winddown_slugs=["trend"],
    )
    assert good.ok  # DBC now counted as expected


def test_reconciliation_passes_when_orphan_flat(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_strategy_positions(data, date(2026, 5, 26), "defensive-etf-allocation", {"SPY": 10})
    write_strategy_positions(data, date(2026, 5, 27), "trend", {"DBC": 0})  # flattened snapshot
    alpaca = [
        PositionRow(
            symbol="SPY",
            qty=10,
            avg_entry_price=500.0,
            market_value=5000.0,
            unrealized_pl=0.0,
            current_price=500.0,
            side="long",
        ),
    ]
    res = check_reconciliation(
        data_dir=data,
        alpaca_positions=alpaca,
        enabled_slugs=["defensive-etf-allocation"],
        winddown_slugs=["trend"],
    )
    assert res.ok


def test_bar_freshness_too_old(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-01")], name="timestamp"),
    )
    df.to_parquet(raw / "SPY.parquet")
    res = check_bar_freshness(
        tmp_path / "data", symbols=["SPY"], asof=date(2026, 5, 25), max_age_days=4
    )
    assert not res.ok
    assert "days old" in res.detail
