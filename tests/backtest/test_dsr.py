from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.dsr import _sr_period_from_returns, deflated_sharpe, probabilistic_sharpe


def _normal_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_per_period_sharpe_helper_matches_mean_over_std() -> None:
    r = _normal_returns(2000, mean=0.001, std=0.01)
    sr = _sr_period_from_returns(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1), rel=1e-9)


def test_psr_returns_high_prob_for_strong_track_record() -> None:
    r = _normal_returns(2000, mean=0.002, std=0.01)  # SR_period ~ 0.2
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    assert psr > 0.99


def test_psr_near_half_when_sharpe_equals_benchmark() -> None:
    r = _normal_returns(2000, mean=0.0, std=0.01)
    sr = _sr_period_from_returns(r)
    psr = probabilistic_sharpe(r, sr_benchmark=sr)
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_returns_zero_on_empty_series() -> None:
    assert probabilistic_sharpe(pd.Series(dtype=float), sr_benchmark=0.0) == 0.0


def test_psr_returns_zero_on_zero_volatility() -> None:
    r = pd.Series([0.0] * 100, index=pd.bdate_range("2010-01-01", periods=100))
    assert probabilistic_sharpe(r, sr_benchmark=0.0) == 0.0


def test_dsr_deflates_strong_in_sample_sharpe_after_many_trials() -> None:
    r = _normal_returns(2000, mean=0.002, std=0.01)
    trial_sharpes = np.linspace(-0.1, 0.2, 50)  # 50 grid trials, max ~0.2 (annualized: high)
    dsr = deflated_sharpe(r, trial_sharpes=trial_sharpes)
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    assert dsr < psr  # multiple-testing correction must reduce the probability


def test_dsr_with_one_trial_equals_psr_at_zero() -> None:
    r = _normal_returns(2000, mean=0.001, std=0.01)
    dsr = deflated_sharpe(r, trial_sharpes=np.array([_sr_period_from_returns(r)]))
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    # n_trials=1 + zero variance => DSR == PSR(0) exactly
    assert dsr == pytest.approx(psr, abs=1e-9)


def test_dsr_returns_zero_on_empty_inputs() -> None:
    assert deflated_sharpe(pd.Series(dtype=float), trial_sharpes=np.array([])) == 0.0
