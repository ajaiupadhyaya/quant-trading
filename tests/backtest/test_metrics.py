"""Tests for quant.backtest.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)


@pytest.fixture
def flat_zero_returns() -> pd.Series:
    return pd.Series(np.zeros(252), index=pd.bdate_range("2024-01-01", periods=252))


@pytest.fixture
def constant_positive_returns() -> pd.Series:
    """Constant +0.1% / day for 252 trading days → ~+28.6% annual, vol=0."""
    return pd.Series(np.full(252, 0.001), index=pd.bdate_range("2024-01-01", periods=252))


@pytest.fixture
def alternating_returns() -> pd.Series:
    """+1%, -1%, +1%, -1%, ... — for win-rate and drawdown tests."""
    vals = np.array([0.01, -0.01] * 126)
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=252))


def test_total_return_zero(flat_zero_returns: pd.Series) -> None:
    assert total_return(flat_zero_returns) == pytest.approx(0.0)


def test_total_return_compounds(constant_positive_returns: pd.Series) -> None:
    expected = (1.001**252) - 1
    assert total_return(constant_positive_returns) == pytest.approx(expected, rel=1e-6)


def test_cagr_handles_subyear(constant_positive_returns: pd.Series) -> None:
    # 252 trading days ≈ 1 calendar year of returns → CAGR ≈ total return.
    assert cagr(constant_positive_returns) == pytest.approx(
        total_return(constant_positive_returns), rel=1e-2
    )


def test_sharpe_zero_vol_returns_zero(constant_positive_returns: pd.Series) -> None:
    # Sharpe undefined when vol == 0; we return 0.0 by convention.
    assert sharpe(constant_positive_returns) == 0.0


def test_sharpe_positive_for_positive_mean() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=pd.bdate_range("2024-01-01", periods=252))
    assert sharpe(r) > 0


def test_sortino_only_penalizes_downside() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=pd.bdate_range("2024-01-01", periods=252))
    # Sortino vs Sharpe: same numerator, downside-only denominator → larger or equal.
    assert sortino(r) >= sharpe(r) - 1e-9


def test_max_drawdown_simple(alternating_returns: pd.Series) -> None:
    # After +1% then -1%, equity = 1.01 * 0.99 = 0.9999 → DD ≈ 1.01 → 0.9999 ≈ -0.0099
    dd = max_drawdown(alternating_returns)
    assert dd < 0
    assert dd > -0.5  # Sanity bound — won't be huge for this series


def test_max_drawdown_zero_for_monotone_up(constant_positive_returns: pd.Series) -> None:
    # All-positive returns → equity is monotone-increasing → max DD = 0.
    assert max_drawdown(constant_positive_returns) == pytest.approx(0.0, abs=1e-9)


def test_win_rate(alternating_returns: pd.Series) -> None:
    assert win_rate(alternating_returns) == pytest.approx(0.5)


def test_win_rate_excludes_zero_returns(flat_zero_returns: pd.Series) -> None:
    # Convention: zero returns are excluded from the denominator → undefined → 0.0.
    assert win_rate(flat_zero_returns) == 0.0


def test_metrics_handle_empty_series() -> None:
    empty = pd.Series(dtype=float)
    assert total_return(empty) == 0.0
    assert sharpe(empty) == 0.0
    assert sortino(empty) == 0.0
    assert max_drawdown(empty) == 0.0
    assert cagr(empty) == 0.0
    assert win_rate(empty) == 0.0
