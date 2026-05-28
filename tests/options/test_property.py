import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.options import HedgeConfig, apply_hedge


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(60, 300),
    trunc=st.integers(40, 59),
)
def test_truncation_invariance(seed, n, trunc):
    rng = np.random.default_rng(seed)
    book = rng.normal(0.0003, 0.012, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    spy_close = pd.Series(100.0 * np.cumprod(1 + book), index=idx)
    returns = pd.Series(book, index=idx)
    cfg = HedgeConfig(use_regime=False)
    full, _ = apply_hedge(returns, spy_close, cfg)
    part, _ = apply_hedge(returns.iloc[:trunc], spy_close.iloc[:trunc], cfg)
    np.testing.assert_allclose(full.iloc[:trunc].to_numpy(), part.to_numpy(), atol=0.0)
