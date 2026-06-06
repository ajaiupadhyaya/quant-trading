"""Tests for quant.execution.policy — impact-aware live execution policy.

TDD: written before the implementation. The policy adjusts NETTED orders at
submission time. Its overriding contract is that ``enabled=False`` reproduces
today's behavior byte-for-byte, and that it can NEVER block a trade because ADV
is unknown (fail-open).
"""

from __future__ import annotations

import math

import pytest

from quant.execution.orders import OrderSide, OrderTemplate, OrderType
from quant.execution.policy import (
    ExecutionPolicyConfig,
    apply_execution_policy,
    cap_qty_to_participation,
    marketable_limit_price,
    participation,
)


def _order(symbol: str, qty: int, side: OrderSide = OrderSide.BUY) -> OrderTemplate:
    return OrderTemplate(symbol=symbol, qty=qty, side=side, strategy_slug="momentum")


# --- participation() ---------------------------------------------------------


def test_participation_basic() -> None:
    assert participation(10_000.0, 100_000.0) == pytest.approx(0.10)


@pytest.mark.parametrize("adv", [0.0, -1.0, float("nan"), float("inf")])
def test_participation_unknown_adv_is_none(adv: float) -> None:
    assert participation(10_000.0, adv) is None


@pytest.mark.parametrize("notional", [float("nan"), float("inf")])
def test_participation_bad_notional_is_none(notional: float) -> None:
    assert participation(notional, 100_000.0) is None


# --- cap_qty_to_participation() ----------------------------------------------

_CFG = ExecutionPolicyConfig(enabled=True, max_participation=0.10)


def test_cap_under_limit_unchanged() -> None:
    # 10 sh * $100 = $1k notional vs 10% of $1M ADV = $100k cap -> uncapped
    capped, deferred = cap_qty_to_participation(10, 100.0, 1_000_000.0, _CFG)
    assert (capped, deferred) == (10, 0)


def test_cap_over_limit_reduces_to_floor() -> None:
    # cap = floor(0.10 * 100_000 / 100) = floor(100) = 100 shares
    capped, deferred = cap_qty_to_participation(250, 100.0, 100_000.0, _CFG)
    assert capped == 100
    assert deferred == 150


def test_cap_reduces_to_zero_when_too_thin() -> None:
    # 10% of $500 ADV = $50 cap; one $100 share doesn't fit -> 0
    capped, deferred = cap_qty_to_participation(3, 100.0, 500.0, _CFG)
    assert capped == 0
    assert deferred == 3


@pytest.mark.parametrize("adv", [0.0, -1.0, float("nan")])
def test_cap_unknown_adv_passes_through(adv: float) -> None:
    capped, deferred = cap_qty_to_participation(250, 100.0, adv, _CFG)
    assert (capped, deferred) == (250, 0)


@pytest.mark.parametrize("ref", [0.0, -1.0, float("nan")])
def test_cap_unknown_ref_passes_through(ref: float) -> None:
    capped, deferred = cap_qty_to_participation(250, ref, 100_000.0, _CFG)
    assert (capped, deferred) == (250, 0)


def test_cap_conserves_quantity_property() -> None:
    for qty in (1, 7, 100, 999):
        capped, deferred = cap_qty_to_participation(qty, 50.0, 30_000.0, _CFG)
        assert capped + deferred == qty
        assert capped >= 0 and deferred >= 0


# --- marketable_limit_price() ------------------------------------------------


def test_marketable_limit_none_when_unconfigured() -> None:
    cfg = ExecutionPolicyConfig(enabled=True, marketable_limit_bps=None)
    assert marketable_limit_price(OrderSide.BUY, 100.0, cfg) is None


def test_marketable_limit_buy_is_above_sell_is_below() -> None:
    cfg = ExecutionPolicyConfig(enabled=True, marketable_limit_bps=20.0)  # 20bps
    buy = marketable_limit_price(OrderSide.BUY, 100.0, cfg)
    sell = marketable_limit_price(OrderSide.SELL, 100.0, cfg)
    assert buy is not None and sell is not None
    assert buy == pytest.approx(100.20, abs=1e-9)
    assert sell == pytest.approx(99.80, abs=1e-9)


# --- apply_execution_policy() : the only function rebalance calls ------------


def test_disabled_is_identity_same_objects() -> None:
    orders = [_order("SPY", 10), _order("TLT", 5, OrderSide.SELL)]
    cfg = ExecutionPolicyConfig(enabled=False)
    out, rows = apply_execution_policy(
        orders, dollar_adv={"SPY": 1.0}, reference_prices={"SPY": 1.0}, cfg=cfg
    )
    assert out == orders  # field-identical
    assert rows == []


def test_enabled_but_adv_unknown_passes_through() -> None:
    orders = [_order("ILLIQ", 1000)]
    cfg = ExecutionPolicyConfig(enabled=True, max_participation=0.10)
    out, _rows = apply_execution_policy(
        orders, dollar_adv={}, reference_prices={"ILLIQ": 100.0}, cfg=cfg
    )
    assert len(out) == 1
    assert out[0].qty == 1000
    assert out[0].order_type is OrderType.MARKET
    assert out[0].limit_price is None


def test_enabled_caps_oversized_order() -> None:
    orders = [_order("THIN", 250)]
    cfg = ExecutionPolicyConfig(enabled=True, max_participation=0.10)
    out, rows = apply_execution_policy(
        orders, dollar_adv={"THIN": 100_000.0}, reference_prices={"THIN": 100.0}, cfg=cfg
    )
    assert len(out) == 1
    assert out[0].qty == 100  # floor(0.10 * 100k / 100)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "THIN"
    assert rows[0]["original_qty"] == 250
    assert rows[0]["capped_qty"] == 100
    assert rows[0]["deferred_qty"] == 150


def test_enabled_drops_fully_deferred_order() -> None:
    orders = [_order("THIN", 3), _order("SPY", 10)]
    cfg = ExecutionPolicyConfig(enabled=True, max_participation=0.10)
    out, rows = apply_execution_policy(
        orders,
        dollar_adv={"THIN": 500.0, "SPY": 100_000_000.0},
        reference_prices={"THIN": 100.0, "SPY": 100.0},
        cfg=cfg,
    )
    out_syms = {o.symbol for o in out}
    assert "THIN" not in out_syms  # fully deferred -> dropped this session
    assert "SPY" in out_syms
    thin_row = next(r for r in rows if r["symbol"] == "THIN")
    assert thin_row["capped_qty"] == 0
    assert thin_row["deferred_qty"] == 3


def test_marketable_limit_applied_above_threshold() -> None:
    # participation of the submitted order is 0.06 (> 0.05 threshold) -> LIMIT
    orders = [_order("MID", 600)]
    cfg = ExecutionPolicyConfig(
        enabled=True,
        max_participation=0.10,
        marketable_limit_bps=15.0,
        marketable_threshold=0.05,
    )
    out, _ = apply_execution_policy(
        orders, dollar_adv={"MID": 1_000_000.0}, reference_prices={"MID": 100.0}, cfg=cfg
    )
    assert out[0].order_type is OrderType.LIMIT
    assert out[0].limit_price == pytest.approx(100.15, abs=1e-9)


def test_marketable_limit_not_applied_below_threshold() -> None:
    orders = [_order("LIQ", 10)]
    cfg = ExecutionPolicyConfig(
        enabled=True,
        max_participation=0.10,
        marketable_limit_bps=15.0,
        marketable_threshold=0.05,
    )
    out, _ = apply_execution_policy(
        orders, dollar_adv={"LIQ": 1_000_000.0}, reference_prices={"LIQ": 100.0}, cfg=cfg
    )
    assert out[0].order_type is OrderType.MARKET
    assert out[0].limit_price is None


def test_side_preserved_through_capping() -> None:
    orders = [_order("THIN", 250, OrderSide.SELL)]
    cfg = ExecutionPolicyConfig(enabled=True, max_participation=0.10)
    out, _ = apply_execution_policy(
        orders, dollar_adv={"THIN": 100_000.0}, reference_prices={"THIN": 100.0}, cfg=cfg
    )
    assert out[0].side is OrderSide.SELL
    assert out[0].qty == 100


def test_plan_rows_participation_is_finite_when_recorded() -> None:
    orders = [_order("THIN", 250)]
    cfg = ExecutionPolicyConfig(enabled=True, max_participation=0.10)
    _, rows = apply_execution_policy(
        orders, dollar_adv={"THIN": 100_000.0}, reference_prices={"THIN": 100.0}, cfg=cfg
    )
    assert rows[0]["participation"] is not None
    assert math.isfinite(rows[0]["participation"])
