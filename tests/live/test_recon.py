"""Tests for quant/live/recon.py — pure logic, no I/O."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import pandas as pd

from quant.execution.alpaca import OrderRow
from quant.live.recon import ReconciliationReport, ReconRow, reconcile


def _trade(
    *,
    coid: str = "trend-20260526-SPY-deadbeef",
    strategy: str = "trend",
    symbol: str = "SPY",
    side: str = "buy",
    qty: int = 100,
    dt: date = date(2026, 5, 26),
) -> dict[str, object]:
    return {
        "date": dt,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "client_order_id": coid,
        "dry_run": False,
    }


def _order(
    *,
    coid: str = "trend-20260526-SPY-deadbeef",
    symbol: str = "SPY",
    side: str = "buy",
    submitted_qty: int = 100,
    filled_qty: int = 100,
    filled_avg_price: float | None = 500.12,
    submitted_at: datetime = datetime(2026, 5, 26, 19, 55, tzinfo=UTC),
    filled_at: datetime | None = datetime(2026, 5, 26, 19, 55, 4, tzinfo=UTC),
    status: str = "filled",
) -> OrderRow:
    return OrderRow(
        client_order_id=coid,
        symbol=symbol,
        side=side,
        submitted_qty=submitted_qty,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        submitted_at=submitted_at,
        filled_at=filled_at,
        status=status,
    )


def _bar_fetcher_for(prices: dict[tuple[str, date], float]) -> Callable[[str, date], float | None]:
    """Return a fake bar fetcher that returns the close price for (symbol, prior_trading_day)."""

    def fetch(symbol: str, prior_trading_day: date) -> float | None:
        return prices.get((symbol, prior_trading_day))

    return fetch


def test_prior_trading_day_alias_skips_memorial_day_2026() -> None:
    from quant.live.recon import prior_trading_day
    assert prior_trading_day(date(2026, 5, 26)) == date(2026, 5, 22)


def test_reconcile_clean_one_to_one_buy() -> None:
    trades = pd.DataFrame([_trade()])
    orders = [_order()]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})  # prior trading day

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
    )

    assert isinstance(report, ReconciliationReport)
    assert report.modeled_slippage_bps == 5.0
    assert len(report.rows) == 1
    row = report.rows[0]
    assert isinstance(row, ReconRow)
    assert row.status == "filled"
    assert row.signal_price == 499.87
    assert row.fill_price == 500.12
    # Buy: (500.12 - 499.87) / 499.87 * 1e4 ≈ 5.0014 bps
    assert row.slippage_bps is not None
    assert abs(row.slippage_bps - 5.001) < 0.01
    assert row.fill_lag_seconds == 4.0


def test_reconcile_clean_one_to_one_sell_signed_correctly() -> None:
    trades = pd.DataFrame([_trade(side="sell")])
    orders = [_order(side="sell", filled_avg_price=499.62)]  # received less than 499.87
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades,
        orders=orders,
        bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26),
        until=date(2026, 5, 26),
    )

    row = report.rows[0]
    # Sell: (499.87 - 499.62) / 499.87 * 1e4 ≈ 5.001 bps (positive = received less)
    assert row.slippage_bps is not None
    assert abs(row.slippage_bps - 5.001) < 0.01


def test_reconcile_missing_order() -> None:
    trades = pd.DataFrame([_trade()])
    orders: list[OrderRow] = []  # nothing came back from Alpaca
    bars = _bar_fetcher_for({})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "missing"
    assert row.filled_qty == 0
    assert row.signal_price is None
    assert row.slippage_bps is None


def test_reconcile_rejected_order() -> None:
    trades = pd.DataFrame([_trade()])
    orders = [_order(filled_qty=0, filled_avg_price=None, filled_at=None, status="rejected")]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "rejected"
    assert row.slippage_bps is None


def test_reconcile_partial_fill_computes_slippage_on_filled_portion() -> None:
    trades = pd.DataFrame([_trade(qty=100)])
    orders = [_order(submitted_qty=100, filled_qty=60, filled_avg_price=500.12, status="partially_filled")]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "partial"
    assert row.filled_qty == 60
    assert row.slippage_bps is not None
    # Buy: (500.12 - 499.87) / 499.87 * 1e4 ≈ 5.001 bps
    assert abs(row.slippage_bps - 5.001) < 0.01


def test_reconcile_no_signal_price_marks_row() -> None:
    trades = pd.DataFrame([_trade()])
    orders = [_order()]
    bars = _bar_fetcher_for({})  # bar fetch returns None

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    row = report.rows[0]
    assert row.status == "no_signal_price"
    assert row.signal_price is None
    assert row.fill_price == 500.12  # still recorded
    assert row.slippage_bps is None


def test_report_aggregate_by_strategy() -> None:
    trades = pd.DataFrame([
        _trade(coid="trend-20260526-SPY-a", strategy="trend", symbol="SPY", qty=100),
        _trade(coid="trend-20260526-DBC-b", strategy="trend", symbol="DBC", qty=50),
        _trade(coid="momentum-20260526-EFA-c", strategy="momentum", symbol="EFA", qty=80),
    ])
    orders = [
        _order(coid="trend-20260526-SPY-a", symbol="SPY", submitted_qty=100, filled_qty=100, filled_avg_price=500.12),
        _order(coid="trend-20260526-DBC-b", symbol="DBC", submitted_qty=50, filled_qty=50, filled_avg_price=25.05),
        _order(coid="momentum-20260526-EFA-c", symbol="EFA", submitted_qty=80, filled_qty=80, filled_avg_price=80.20),
    ]
    bars = _bar_fetcher_for({
        ("SPY", date(2026, 5, 22)): 499.87,
        ("DBC", date(2026, 5, 22)): 25.00,
        ("EFA", date(2026, 5, 22)): 80.00,
    })

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    by_strategy = report.aggregate_by_strategy()
    assert set(by_strategy.keys()) == {"trend", "momentum"}
    assert by_strategy["trend"]["n_filled"] == 2
    assert by_strategy["momentum"]["n_filled"] == 1
    # trend mean slippage = mean of ~5 bps and 20 bps = ~12.5
    assert by_strategy["trend"]["mean_slippage_bps"] is not None
    assert 10 < by_strategy["trend"]["mean_slippage_bps"] < 15


def test_report_aggregate_by_symbol() -> None:
    trades = pd.DataFrame([
        _trade(coid="trend-20260526-SPY-a", symbol="SPY", qty=100),
        _trade(coid="trend-20260526-SPY-b", symbol="SPY", qty=50),
    ])
    orders = [
        _order(coid="trend-20260526-SPY-a", symbol="SPY", submitted_qty=100, filled_qty=100, filled_avg_price=500.12),
        _order(coid="trend-20260526-SPY-b", symbol="SPY", submitted_qty=50, filled_qty=50, filled_avg_price=500.50),
    ]
    bars = _bar_fetcher_for({("SPY", date(2026, 5, 22)): 499.87})

    report = reconcile(
        trades=trades, orders=orders, bar_fetcher=bars,
        modeled_slippage_bps=5.0,
        since=date(2026, 5, 26), until=date(2026, 5, 26),
    )

    by_symbol = report.aggregate_by_symbol()
    assert set(by_symbol.keys()) == {"SPY"}
    assert by_symbol["SPY"]["n_filled"] == 2
    assert by_symbol["SPY"]["mean_slippage_bps"] is not None
