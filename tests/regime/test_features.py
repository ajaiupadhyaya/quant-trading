from __future__ import annotations

import numpy as np
import pandas as pd

from quant.regime.features import FeatureConfig, _extract_close, build_feature_matrix


def _series(n: int, seed: int) -> pd.Series:
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_feature_matrix_columns_and_no_nan_tail():
    spy = _series(400, 1)
    vix = pd.Series(np.full(400, 18.0), index=spy.index)
    cfg = FeatureConfig(use_term_spread=False)
    feats = build_feature_matrix(spy_close=spy, vix=vix, dgs10=None, dgs2=None, config=cfg)
    assert list(feats.columns) == ["ret", "vol", "vix", "drawdown"]
    # The warmup window is dropped; remaining rows are fully populated.
    assert not feats.isna().any().any()
    assert len(feats) > 0


def test_standardization_is_trailing_only():
    # Build features over the full series, then over a truncated prefix.
    # A given date's standardized features must be identical in both — i.e.
    # standardization uses only trailing data, never the full sample.
    spy = _series(400, 2)
    vix = pd.Series(np.full(400, 18.0), index=spy.index)
    cfg = FeatureConfig(use_term_spread=False, standardize_window=120)
    full = build_feature_matrix(spy_close=spy, vix=vix, dgs10=None, dgs2=None, config=cfg)
    prefix = build_feature_matrix(
        spy_close=spy.iloc[:300], vix=vix.iloc[:300], dgs10=None, dgs2=None, config=cfg
    )
    shared = prefix.index
    pd.testing.assert_frame_equal(full.loc[shared], prefix, atol=1e-9)


def test_extract_close_multiindex():
    # Simulate the exact shape get_bars returns: pd.concat({symbol: df}, axis=1)
    # which always produces a (symbol, field) MultiIndex column frame.
    idx = pd.bdate_range("2020-01-01", periods=5)
    per_symbol = pd.DataFrame(
        {"open": [1.0] * 5, "high": [2.0] * 5, "low": [0.5] * 5, "close": [1.5] * 5},
        index=idx,
    )
    wide = pd.concat({"SPY": per_symbol}, axis=1)  # MultiIndex (SPY, field)
    assert isinstance(wide.columns, pd.MultiIndex)
    close = _extract_close(wide, "SPY")
    assert list(close) == [1.5] * 5
    assert close.index.equals(idx)


def test_extract_close_flat():
    # Edge case: flat "close" column (not from get_bars, but defensively supported).
    idx = pd.bdate_range("2020-01-01", periods=3)
    flat = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx)
    close = _extract_close(flat, "SPY")
    assert list(close) == [10.0, 11.0, 12.0]
