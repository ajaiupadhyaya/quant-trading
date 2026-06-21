"""Portfolio leverage knob: normalization, hard cap, fail-safe."""

from __future__ import annotations

import pytest

from quant.live.rebalance import _leveraged_weights


def test_normalizes_to_target_gross() -> None:
    # allocation summing to 0.8 -> deployed at exactly target gross.
    w = _leveraged_weights({"a": 0.4, "b": 0.4}, 1.0)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["a"] == pytest.approx(0.5) and w["b"] == pytest.approx(0.5)


def test_one_point_five_x() -> None:
    w = _leveraged_weights({"a": 0.4, "b": 0.4}, 1.5)
    assert sum(w.values()) == pytest.approx(1.5)
    assert w["a"] == pytest.approx(0.75)


def test_hard_cap_at_two() -> None:
    w = _leveraged_weights({"a": 0.5, "b": 0.5}, 10.0)  # fat-finger
    assert sum(w.values()) == pytest.approx(2.0)


def test_clamped_at_zero() -> None:
    w = _leveraged_weights({"a": 0.5, "b": 0.5}, -3.0)
    assert sum(w.values()) == pytest.approx(0.0)


def test_preserves_relative_weights() -> None:
    w = _leveraged_weights({"a": 0.6, "b": 0.2}, 1.0)  # 3:1 split
    assert w["a"] == pytest.approx(0.75) and w["b"] == pytest.approx(0.25)


def test_zero_sum_allocation_is_safe() -> None:
    alloc = {"a": 0.0, "b": 0.0}
    assert _leveraged_weights(alloc, 1.5) == alloc
