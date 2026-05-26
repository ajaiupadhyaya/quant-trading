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
