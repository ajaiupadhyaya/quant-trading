"""Unit tests for the free-function signal wrappers with hand-checkable inputs."""

from __future__ import annotations

import numpy as np

from quant.research.signals import (
    breadth,
    drawdown,
    momentum,
    realized_vol,
    rsi,
    trend_filter,
)
from tests.research.conftest import close_panel, constant, falling, peak_then_drop, rising


def test_momentum_rising_is_positive() -> None:
    m = momentum(rising(), lookbacks=(63, 126))
    assert m["mom_63"].iloc[-1] > 0
    assert m["mom_126"].iloc[-1] > 0


def test_momentum_falling_is_negative() -> None:
    m = momentum(falling(), lookbacks=(63,))
    assert m["mom_63"].iloc[-1] < 0


def test_momentum_insufficient_history_is_nan() -> None:
    m = momentum(rising(n=40), lookbacks=(63,))
    assert np.isnan(m["mom_63"].iloc[-1])  # only 40 obs, 63d lookback unavailable


def test_breadth_all_rising_is_one() -> None:
    panel = close_panel(seed=1)
    # Force every column strictly increasing so all sit above their 200d MA.
    rising_panel = panel.copy()
    for c in rising_panel.columns:
        rising_panel[c] = np.linspace(50.0, 150.0, len(rising_panel))
    assert breadth(rising_panel, ma_days=200).iloc[-1] == 1.0


def test_breadth_all_falling_is_zero() -> None:
    panel = close_panel(seed=2)
    falling_panel = panel.copy()
    for c in falling_panel.columns:
        falling_panel[c] = np.linspace(150.0, 50.0, len(falling_panel))
    assert breadth(falling_panel, ma_days=200).iloc[-1] == 0.0


def test_trend_filter_above_and_below() -> None:
    assert bool(trend_filter(rising(), ma_days=200).iloc[-1]) is True
    assert bool(trend_filter(falling(), ma_days=200).iloc[-1]) is False


def test_realized_vol_constant_is_zero_not_nan() -> None:
    rv = realized_vol(constant(), window=21)
    assert rv.iloc[-1] == 0.0  # zero variance, defined (not NaN), after warmup


def test_realized_vol_positive_for_noisy_series() -> None:
    rng = np.random.default_rng(3)
    s = rising().copy()
    s.iloc[:] = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(s))))
    rv = realized_vol(s, window=21, annualize=True)
    assert rv.iloc[-1] > 0.0 and np.isfinite(rv.iloc[-1])


def test_rsi_all_gains_is_100() -> None:
    assert rsi(rising(), period=14).iloc[-1] == 100.0


def test_rsi_all_losses_is_0() -> None:
    assert rsi(falling(), period=14).iloc[-1] == 0.0


def test_rsi_oscillating_is_near_50() -> None:
    idx = rising().index
    osc = 100.0 + np.where(np.arange(len(idx)) % 2 == 0, 0.0, 1.0)
    import pandas as pd

    s = pd.Series(osc, index=idx)
    r = rsi(s, period=14).iloc[-1]
    assert 35.0 < r < 65.0  # equal up/down magnitude -> RS ~ 1 -> RSI ~ 50


def test_drawdown_new_high_is_zero() -> None:
    assert drawdown(rising()).iloc[-1] == 0.0  # monotone up -> always at the peak


def test_drawdown_minus_20pct() -> None:
    dd = drawdown(peak_then_drop(drop=0.20)).iloc[-1]
    assert abs(dd - (-0.20)) < 1e-6
