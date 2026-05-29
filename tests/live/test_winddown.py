"""Tests for the orphan wind-down helpers."""

from __future__ import annotations

from quant.live.winddown import capped_qty


def test_cap_binds_when_order_exceeds_participation():
    # ADV $1,000,000, 10% => $100,000 budget; price $100 => 1000 shares max.
    assert capped_qty(5000, 1_000_000.0, 100.0, 0.10) == 1000


def test_cap_passes_through_when_order_within_budget():
    assert capped_qty(200, 1_000_000.0, 100.0, 0.10) == 200


def test_zero_or_negative_adv_is_zero():
    assert capped_qty(500, 0.0, 100.0, 0.10) == 0
    assert capped_qty(500, -1.0, 100.0, 0.10) == 0


def test_nonpositive_price_is_zero():
    assert capped_qty(500, 1_000_000.0, 0.0, 0.10) == 0


def test_nonfinite_inputs_zero():
    assert capped_qty(500, float("nan"), 100.0, 0.10) == 0
    assert capped_qty(500, 1_000_000.0, float("inf"), 0.10) == 0


def test_nonpositive_order_qty_is_zero():
    assert capped_qty(0, 1_000_000.0, 100.0, 0.10) == 0
