"""Tests for the pure market-impact model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.impact import market_impact_bps, trailing_dollar_adv


def _bars(symbol: str, closes: list[float], volumes: list[int]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.array(volumes, dtype=np.int64),
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return pd.concat({symbol: df}, axis=1)


# ---- market_impact_bps ----


def test_participation_one_returns_coef():
    assert market_impact_bps(1_000_000.0, 1_000_000.0, 100.0) == pytest.approx(100.0)


def test_participation_quarter_is_half_coef():
    assert market_impact_bps(250_000.0, 1_000_000.0, 100.0) == pytest.approx(50.0)


def test_impact_is_concave_in_size():
    small = market_impact_bps(1_000_000.0, 1_000_000_000.0, 100.0)
    big = market_impact_bps(2_000_000.0, 1_000_000_000.0, 100.0)
    assert big < 2.0 * small


def test_nonpositive_adv_is_zero():
    assert market_impact_bps(1_000_000.0, 0.0, 100.0) == 0.0
    assert market_impact_bps(1_000_000.0, -5.0, 100.0) == 0.0


def test_nonpositive_notional_is_zero():
    assert market_impact_bps(0.0, 1_000_000.0, 100.0) == 0.0
    assert market_impact_bps(-10.0, 1_000_000.0, 100.0) == 0.0


def test_nonfinite_inputs_are_zero():
    assert market_impact_bps(float("nan"), 1_000_000.0, 100.0) == 0.0
    assert market_impact_bps(1_000_000.0, float("inf"), 100.0) == 0.0


# ---- trailing_dollar_adv ----


def test_adv_mean_over_strictly_prior_window():
    bars = _bars("AAA", [10.0, 10.0, 10.0], [100, 200, 300])
    fill_ts = bars.index[2]
    assert trailing_dollar_adv(bars, "AAA", fill_ts, window=5) == pytest.approx(1500.0)


def test_adv_excludes_fill_bar_volume_pit():
    bars = _bars("AAA", [10.0, 10.0, 10.0], [100, 200, 10_000_000])
    fill_ts = bars.index[2]
    assert trailing_dollar_adv(bars, "AAA", fill_ts, window=5) == pytest.approx(1500.0)


def test_adv_respects_window_length():
    bars = _bars("AAA", [10.0, 10.0, 10.0, 10.0], [100, 200, 300, 400])
    fill_ts = bars.index[3]
    assert trailing_dollar_adv(bars, "AAA", fill_ts, window=2) == pytest.approx(2500.0)


def test_adv_no_prior_history_is_zero():
    bars = _bars("AAA", [10.0, 10.0], [100, 200])
    assert trailing_dollar_adv(bars, "AAA", bars.index[0], window=5) == 0.0


def test_adv_missing_symbol_is_zero():
    bars = _bars("AAA", [10.0, 10.0], [100, 200])
    assert trailing_dollar_adv(bars, "ZZZ", bars.index[1], window=5) == 0.0
