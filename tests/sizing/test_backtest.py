from __future__ import annotations

import numpy as np
import pandas as pd

from quant.sizing.backtest import SizingComparison, apply_sizing, compare_sizing
from quant.sizing.models import SizingConfig


def _returns(n: int = 400, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx, name="returns")


def test_apply_sizing_shape_and_index() -> None:
    r = _returns()
    sized, gross = apply_sizing(r, SizingConfig())
    assert len(sized) == len(r)
    assert len(gross) == len(r)
    assert (sized.index == r.index).all()
    assert (gross.index == r.index).all()


def test_pit_truncation_invariance() -> None:
    # THE critical test: gross[:k] computed on full series must equal gross
    # computed on the truncated series. No look-ahead.
    r = _returns(n=500, seed=4)
    cfg = SizingConfig()
    _, gross_full = apply_sizing(r, cfg)
    k = 300
    _, gross_trunc = apply_sizing(r.iloc[:k], cfg)
    np.testing.assert_allclose(
        gross_full.iloc[:k].to_numpy(), gross_trunc.to_numpy(), rtol=0, atol=0
    )


def test_gross_uses_only_prior_returns() -> None:
    # Day 0 gross must be the all-neutral default regardless of r[0]'s value,
    # because history before day 0 is empty.
    r = _returns(n=50)
    cfg = SizingConfig()
    _, gross = apply_sizing(r, cfg)
    # empty history -> vol/kelly/drawdown neutral; regime None -> 1.0
    assert gross.iloc[0] == 1.0


def test_regime_label_is_as_of_yesterday() -> None:
    r = _returns(n=10)
    # crisis from day 5 onward; gross on day 5 should still use day 4's label
    labels = pd.Series(["calm-bull"] * 10, index=r.index, name="label")
    labels.iloc[5:] = "crisis"
    cfg = SizingConfig(use_vol_target=False, use_kelly=False, use_drawdown=False, use_regime=True)
    _, gross = apply_sizing(r, cfg, regime_labels=labels)
    # day 5 uses day 4 label (calm-bull -> 1.0); day 6 uses day 5 label (crisis -> 0.0)
    assert gross.iloc[5] == 1.0
    assert gross.iloc[6] == 0.0


def test_sized_returns_equal_gross_times_returns() -> None:
    r = _returns(n=100, seed=2)
    cfg = SizingConfig()
    sized, gross = apply_sizing(r, cfg)
    np.testing.assert_allclose(sized.to_numpy(), (gross * r).to_numpy())


def test_compare_sizing_returns_complete_finite_metrics() -> None:
    r = _returns()
    comp = compare_sizing(r, SizingConfig())
    assert isinstance(comp, SizingComparison)
    keys = {"total_return", "cagr", "sharpe", "sortino", "max_drawdown", "ann_vol", "win_rate"}
    assert set(comp.baseline) == keys
    assert set(comp.sized) == keys
    assert all(np.isfinite(v) for v in comp.baseline.values())
    assert all(np.isfinite(v) for v in comp.sized.values())
    assert np.isfinite(comp.gross_mean)
    assert comp.gross_min <= comp.gross_mean <= comp.gross_max


def test_compare_sizing_empty_returns_is_safe() -> None:
    comp = compare_sizing(pd.Series(dtype=float), SizingConfig())
    assert all(np.isfinite(v) for v in comp.sized.values())
