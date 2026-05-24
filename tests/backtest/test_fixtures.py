"""Sanity tests for the synthetic_bars fixture and EqualWeightStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd

from tests.conftest import EqualWeightStrategy, synthetic_bars


def test_synthetic_bars_shape() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 31))
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert set(bars.columns.get_level_values(0)) == {"AAA", "BBB"}
    assert "close" in bars.columns.get_level_values(1)
    assert len(bars) > 20  # 22-ish business days in Jan 2024


def test_synthetic_bars_deterministic() -> None:
    a = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=7)
    b = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_bars_different_seeds_differ() -> None:
    a = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=1)
    b = synthetic_bars(["AAA"], date(2024, 1, 1), date(2024, 1, 31), seed=2)
    assert not a.equals(b)


def test_equal_weight_target_positions() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 31), seed=0)
    strat = EqualWeightStrategy(bars=bars)
    target = strat.target_positions(date(2024, 1, 5), equity=100_000.0)
    assert set(target.keys()) <= {"AAA", "BBB"}
    assert all(qty > 0 for qty in target.values())
