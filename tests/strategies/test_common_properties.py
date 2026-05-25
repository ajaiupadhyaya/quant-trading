"""Hypothesis property tests for quant.strategies._common helpers.

``size_to_shares`` is the choke point that every concrete strategy uses to
convert a per-symbol weight series into integer share counts. If it ever
silently produces over-allocated dollar exposure, every strategy is broken at
once — so the invariants here matter.
"""

from __future__ import annotations

import math

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.strategies._common import size_to_shares

SYMBOL = st.sampled_from(["AAA", "BBB", "CCC", "DDD"])
POSITIVE_PRICE = st.floats(min_value=0.5, max_value=5000.0, allow_nan=False, allow_infinity=False)
WEIGHT = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
EQUITY = st.floats(min_value=1_000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)


@st.composite
def weight_price_panels(draw: st.DrawFn) -> tuple[pd.Series, pd.Series]:
    n = draw(st.integers(min_value=1, max_value=4))
    symbols = ["S" + str(i) for i in range(n)]
    weights = pd.Series(draw(st.lists(WEIGHT, min_size=n, max_size=n)), index=symbols)
    prices = pd.Series(draw(st.lists(POSITIVE_PRICE, min_size=n, max_size=n)), index=symbols)
    return weights, prices


@given(panel=weight_price_panels(), equity=EQUITY)
@settings(max_examples=300, deadline=None)
def test_shares_never_overshoot_target_notional(
    panel: tuple[pd.Series, pd.Series], equity: float
) -> None:
    """Integer-floor sizing must keep |shares * price| <= |weight * equity| + 1 share-tick."""
    weights, prices = panel
    out = size_to_shares(weights, prices, equity)
    for sym, shares in out.items():
        target_dollars = abs(float(weights.loc[sym]) * equity)
        actual_dollars = abs(shares * float(prices.loc[sym]))
        # Floor sizing means actual <= target; never strictly greater.
        assert actual_dollars <= target_dollars + 1e-6, (
            f"{sym}: target={target_dollars}, actual={actual_dollars}"
        )


@given(panel=weight_price_panels(), equity=EQUITY)
@settings(max_examples=300, deadline=None)
def test_shares_sign_matches_weight_sign(panel: tuple[pd.Series, pd.Series], equity: float) -> None:
    weights, prices = panel
    out = size_to_shares(weights, prices, equity)
    for sym, shares in out.items():
        w = float(weights.loc[sym])
        if shares > 0:
            assert w > 0
        elif shares < 0:
            assert w < 0


@given(panel=weight_price_panels())
@settings(max_examples=100, deadline=None)
def test_zero_equity_emits_no_shares(panel: tuple[pd.Series, pd.Series]) -> None:
    weights, prices = panel
    assert size_to_shares(weights, prices, 0.0) == {}
    assert size_to_shares(weights, prices, -100.0) == {}


@given(panel=weight_price_panels(), equity=EQUITY)
@settings(max_examples=200, deadline=None)
def test_output_keys_subset_of_input_keys(
    panel: tuple[pd.Series, pd.Series], equity: float
) -> None:
    weights, prices = panel
    out = size_to_shares(weights, prices, equity)
    valid_inputs = {str(s) for s in weights.index if math.isfinite(float(weights.loc[s]))}
    assert set(out.keys()).issubset(valid_inputs)
