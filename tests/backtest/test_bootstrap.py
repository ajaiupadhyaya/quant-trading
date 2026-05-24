from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.bootstrap import (
    BootstrapCI,
    bootstrap_ci,
    iid_resample,
    stationary_block_resample,
)


def _normal_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_iid_resample_preserves_length_and_is_a_subset_of_input() -> None:
    r = _normal_returns(500, mean=0.0, std=0.01)
    sample = iid_resample(r, seed=42)
    assert len(sample) == len(r)
    assert set(np.unique(sample.to_numpy())).issubset(set(np.unique(r.to_numpy())))


def test_iid_resample_is_deterministic_under_same_seed() -> None:
    r = _normal_returns(200, 0.0, 0.01)
    s1 = iid_resample(r, seed=7)
    s2 = iid_resample(r, seed=7)
    pd.testing.assert_series_equal(s1, s2)


def test_iid_resample_differs_across_seeds() -> None:
    r = _normal_returns(200, 0.0, 0.01)
    s1 = iid_resample(r, seed=1)
    s2 = iid_resample(r, seed=2)
    assert not s1.equals(s2)


def test_stationary_block_resample_preserves_length() -> None:
    r = _normal_returns(500, 0.0, 0.01)
    s = stationary_block_resample(r, mean_block_len=5, seed=0)
    assert len(s) == len(r)


def test_stationary_block_resample_with_block_len_one_is_iid_like() -> None:
    # mean_block_len=1 collapses to IID; means should be in the same ballpark as IID.
    r = _normal_returns(2000, 0.001, 0.01)
    block = stationary_block_resample(r, mean_block_len=1, seed=0)
    assert block.mean() == pytest.approx(r.mean(), abs=0.003)


def test_bootstrap_ci_returns_bracketing_intervals_for_total_return() -> None:
    r = _normal_returns(1000, mean=0.001, std=0.01)
    ci = bootstrap_ci(r, n_resamples=200, mean_block_len=5, seed=0)
    assert isinstance(ci, BootstrapCI)
    assert ci.total_return_p05 < ci.total_return_median < ci.total_return_p95
    assert ci.sharpe_p05 < ci.sharpe_median < ci.sharpe_p95
    assert ci.max_drawdown_p05 < ci.max_drawdown_p95  # both negative; p05 is "worse"


def test_bootstrap_ci_empty_series_returns_zeros() -> None:
    ci = bootstrap_ci(pd.Series(dtype=float), n_resamples=50, mean_block_len=5, seed=0)
    assert ci.total_return_median == 0.0
    assert ci.sharpe_median == 0.0
    assert ci.max_drawdown_median == 0.0


def test_bootstrap_ci_deterministic_under_seed() -> None:
    r = _normal_returns(400, 0.0005, 0.01)
    a = bootstrap_ci(r, n_resamples=100, mean_block_len=5, seed=11)
    b = bootstrap_ci(r, n_resamples=100, mean_block_len=5, seed=11)
    assert a == b
