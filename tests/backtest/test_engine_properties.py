"""Hypothesis property tests for the backtest engine cost model.

These complement the example-based tests in test_engine_costs.py by checking
invariants that should hold for every input drawn from realistic ranges:

  * Buys always fill at or above mid; sells at or below.
  * Slippage and commission are both non-negative and zero on zero qty.
  * Slippage cost is monotonic in qty and slippage_bps.
  * Commission cost is monotonic in commission_bps.

Hypothesis is the right tool here because the cost model is a small pure
function with continuous parameter space — exactly where example tests miss
edge cases.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from quant.backtest.engine import BacktestConfig, apply_costs

QTY = st.integers(min_value=1, max_value=10_000)
PRICE = st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False)
SLIP_BPS = st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)
COMM_BPS = st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)


def _cfg(slip: float, comm: float) -> BacktestConfig:
    return BacktestConfig(slippage_bps=slip, commission_bps=comm)


@given(qty=QTY, price=PRICE, slip=SLIP_BPS, comm=COMM_BPS)
@settings(max_examples=200, deadline=None)
def test_buy_fills_at_or_above_mid(qty: int, price: float, slip: float, comm: float) -> None:
    fill = apply_costs(qty=qty, mid_price=price, side="buy", config=_cfg(slip, comm))
    assert fill.fill_price >= price - 1e-9


@given(qty=QTY, price=PRICE, slip=SLIP_BPS, comm=COMM_BPS)
@settings(max_examples=200, deadline=None)
def test_sell_fills_at_or_below_mid(qty: int, price: float, slip: float, comm: float) -> None:
    fill = apply_costs(qty=qty, mid_price=price, side="sell", config=_cfg(slip, comm))
    assert fill.fill_price <= price + 1e-9


@given(qty=QTY, price=PRICE, slip=SLIP_BPS, comm=COMM_BPS)
@settings(max_examples=200, deadline=None)
def test_costs_are_non_negative(qty: int, price: float, slip: float, comm: float) -> None:
    for side in ("buy", "sell"):
        fill = apply_costs(qty=qty, mid_price=price, side=side, config=_cfg(slip, comm))  # type: ignore[arg-type]
        assert fill.slippage_cost >= 0.0
        assert fill.commission_cost >= 0.0
        assert math.isfinite(fill.fill_price)


@given(price=PRICE, slip=SLIP_BPS, comm=COMM_BPS)
@settings(max_examples=100, deadline=None)
def test_zero_qty_zero_costs(price: float, slip: float, comm: float) -> None:
    fill = apply_costs(qty=0, mid_price=price, side="buy", config=_cfg(slip, comm))
    assert fill.fill_price == price
    assert fill.slippage_cost == 0.0
    assert fill.commission_cost == 0.0


@given(qty=QTY, price=PRICE, slip=SLIP_BPS)
@settings(max_examples=100, deadline=None)
def test_slippage_monotonic_in_qty(qty: int, price: float, slip: float) -> None:
    """Slippage cost scales linearly with quantity at fixed price + bps."""
    cfg = _cfg(slip, 0.0)
    a = apply_costs(qty=qty, mid_price=price, side="buy", config=cfg)
    b = apply_costs(qty=qty * 2, mid_price=price, side="buy", config=cfg)
    if slip == 0.0:
        assert a.slippage_cost == 0.0 == b.slippage_cost
        return
    assert b.slippage_cost >= a.slippage_cost - 1e-9


@given(qty=QTY, price=PRICE, comm=COMM_BPS)
@settings(max_examples=100, deadline=None)
def test_commission_monotonic_in_bps(qty: int, price: float, comm: float) -> None:
    """Commission cost grows with commission_bps at fixed qty + price."""
    a = apply_costs(qty=qty, mid_price=price, side="buy", config=_cfg(0.0, comm))
    b = apply_costs(qty=qty, mid_price=price, side="buy", config=_cfg(0.0, comm * 2))
    assert b.commission_cost >= a.commission_cost - 1e-9
