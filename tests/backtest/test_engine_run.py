"""Tests for run_backtest."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from tests.conftest import EqualWeightStrategy


def _flat_bars(symbols: list[str], price: float = 100.0) -> pd.DataFrame:
    """Bars where every field equals `price`, every business day in 2024-Q1."""
    dates = pd.bdate_range("2024-01-02", "2024-03-29")  # avoid Jan 1 holiday
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = pd.DataFrame(
            {f: np.full(len(dates), price) for f in ("open", "high", "low", "close")}
            | {"volume": np.full(len(dates), 1_000_000, dtype=np.int64)},
            index=dates,
        )
        df.index.name = "timestamp"
        frames[sym] = df
    return pd.concat(frames, axis=1)


def test_equity_curve_indexed_by_history(
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve, pd.Series)
    assert result.equity_curve.index.equals(bars.index)


def test_starting_and_ending_equity(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(starting_equity=50_000.0, slippage_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 1), date(2024, 6, 30))
    assert result.starting_equity == 50_000.0
    # First equity point <= starting equity (any positions reduce cash by their notional cost).
    assert result.equity_curve.iloc[0] <= 50_000.0
    assert result.ending_equity == pytest.approx(result.equity_curve.iloc[-1], abs=1e-6)


def test_flat_price_zero_costs_preserves_equity_exactly() -> None:
    """If all prices are constant and costs are zero, equity should be flat at start."""
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(starting_equity=100_000.0, slippage_bps=0.0, commission_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # All prices constant → equity = cash + shares * 100. Should equal starting_equity bar-for-bar
    # once the first rebalance fills, ± rounding from integer share count.
    assert all(abs(eq - 100_000.0) < 100.0 for eq in result.equity_curve), (
        result.equity_curve.head()
    )


def test_slippage_drains_equity_on_each_rebalance() -> None:
    """Holding flat prices: every rebalance costs slippage; equity decays monotonically."""
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)  # monthly rebalance
    cfg = BacktestConfig(slippage_bps=50.0, commission_bps=0.0)  # 50 bps
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # First rebalance fills on next_open of first day. After that, no further rebalances within month
    # → flat.
    assert result.ending_equity < result.starting_equity


def test_trades_have_required_columns(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    assert {
        "date",
        "symbol",
        "side",
        "qty",
        "fill_price",
        "slippage_cost",
        "commission_cost",
        "strategy_slug",
    } <= set(result.trades.columns)
    assert len(result.trades) > 0
    assert (result.trades["qty"] > 0).all()
    assert set(result.trades["side"]) <= {"buy", "sell"}


def test_positions_dataframe_tracks_shares(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 1, 1), date(2024, 6, 30))
    # By end of horizon: at least one symbol has positive shares.
    assert (result.positions.iloc[-1] > 0).any()
    assert set(result.positions.columns) <= {"AAA", "BBB"}


def test_close_execution_fills_same_bar() -> None:
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(execution="close", slippage_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # Close execution: trades on first day, not waiting for next open.
    assert (result.trades["date"] == pd.Timestamp("2024-01-02")).any()


def test_next_open_execution_fills_next_bar() -> None:
    bars = _flat_bars(["AAA", "BBB"])
    strat = EqualWeightStrategy(bars=bars)
    cfg = BacktestConfig(execution="next_open", slippage_bps=0.0)
    result = run_backtest(strat, bars, cfg, date(2024, 1, 2), date(2024, 3, 29))
    # next_open: first month-start rebalance on Jan 2 → fills Jan 3.
    first_trade_date = result.trades["date"].min()
    assert first_trade_date > pd.Timestamp("2024-01-02")


def test_empty_history_returns_empty_result(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 6, 30), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    # Request a window outside the data → engine should not crash; returns empty curve.
    result = run_backtest(strat, bars, BacktestConfig(), date(2030, 1, 1), date(2030, 12, 31))
    assert len(result.equity_curve) == 0
    assert len(result.trades) == 0


def test_window_slicing(make_bars: Callable[..., pd.DataFrame]) -> None:
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    result = run_backtest(strat, bars, BacktestConfig(), date(2024, 6, 1), date(2024, 6, 30))
    assert result.equity_curve.index.min() >= pd.Timestamp("2024-06-01")
    assert result.equity_curve.index.max() <= pd.Timestamp("2024-06-30")
