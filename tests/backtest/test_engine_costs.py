"""Tests for the engine cost model."""

from __future__ import annotations

import pytest

from quant.backtest.engine import BacktestConfig, apply_costs


def test_buy_slippage_raises_fill_price() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=0.0)  # 10 bps = 0.10%
    fill = apply_costs(qty=100, mid_price=50.0, side="buy", config=cfg)
    # Buy is hit up by 10 bps: 50.00 * 1.001 = 50.05
    assert fill.fill_price == pytest.approx(50.05, abs=1e-6)
    assert fill.slippage_cost == pytest.approx(100 * (50.05 - 50.0), abs=1e-6)
    assert fill.commission_cost == 0.0


def test_sell_slippage_lowers_fill_price() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=0.0)
    fill = apply_costs(qty=100, mid_price=50.0, side="sell", config=cfg)
    # Sell is hit down by 10 bps: 50.00 * 0.999 = 49.95
    assert fill.fill_price == pytest.approx(49.95, abs=1e-6)
    # Slippage cost is measured as cash lost vs mid: (50.0 - 49.95) * 100
    assert fill.slippage_cost == pytest.approx(100 * (50.0 - 49.95), abs=1e-6)


def test_commission_is_bps_of_notional() -> None:
    cfg = BacktestConfig(slippage_bps=0.0, commission_bps=5.0)  # 5 bps = 0.05%
    fill = apply_costs(qty=100, mid_price=50.0, side="buy", config=cfg)
    notional = 100 * fill.fill_price
    assert fill.commission_cost == pytest.approx(notional * 0.0005, abs=1e-6)


def test_zero_costs_returns_mid() -> None:
    cfg = BacktestConfig(slippage_bps=0.0, commission_bps=0.0)
    fill = apply_costs(qty=100, mid_price=42.0, side="buy", config=cfg)
    assert fill.fill_price == 42.0
    assert fill.slippage_cost == 0.0
    assert fill.commission_cost == 0.0


def test_invalid_side_raises() -> None:
    cfg = BacktestConfig()
    with pytest.raises(ValueError):
        apply_costs(qty=10, mid_price=50.0, side="hold", config=cfg)  # type: ignore[arg-type]


def test_zero_qty_costs_zero() -> None:
    cfg = BacktestConfig(slippage_bps=10.0, commission_bps=10.0)
    fill = apply_costs(qty=0, mid_price=50.0, side="buy", config=cfg)
    assert fill.slippage_cost == 0.0
    assert fill.commission_cost == 0.0


def test_default_config_values() -> None:
    cfg = BacktestConfig()
    assert cfg.starting_equity == 100_000.0
    assert cfg.slippage_bps == 5.0
    assert cfg.commission_bps == 0.0
    assert cfg.execution == "next_open"
    assert cfg.annual_borrow_bps == 50.0
    assert cfg.annual_financing_bps == 200.0
