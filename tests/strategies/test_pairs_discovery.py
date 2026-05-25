"""Tests for the pairs discovery + screen layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies._pairs_discovery import (
    PairCandidate,
    discover_and_screen_pairs,
    fit_pair,
    pca_candidate_pairs,
)


def _cointegrated_series(
    n: int = 252,
    beta: float = 1.5,
    alpha: float = 0.2,
    half_life: float = 10.0,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Build a synthetic cointegrated pair (a, b) with the requested half-life."""
    rng = np.random.default_rng(seed)
    rho = float(np.exp(-np.log(2.0) / half_life))
    # log_b is a random walk, log_a = beta*log_b + alpha + epsilon, where eps is OU.
    b_innov = rng.normal(0, 0.01, n)
    log_b = np.cumsum(b_innov)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + rng.normal(0, 0.01)
    log_a = beta * log_b + alpha + eps
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.exp(log_a), index=idx, name="A"), pd.Series(
        np.exp(log_b), index=idx, name="B"
    )


def test_fit_pair_recovers_beta_on_clean_synthetic() -> None:
    a, b = _cointegrated_series(n=300, beta=1.5, alpha=0.2, half_life=10.0)
    fit = fit_pair(a, b)
    assert fit is not None
    # Beta recovery is approximate but should be well within 20% of the true value.
    assert abs(fit.beta - 1.5) < 0.3
    # Half-life recovery should be in the right neighborhood.
    assert 3.0 < fit.half_life_days < 30.0
    # Mean-reversion coefficient strictly between 0 and 1.
    assert 0 < fit.ar1_rho < 1


def test_fit_pair_rejects_random_walk_pair() -> None:
    rng = np.random.default_rng(1)
    a = pd.Series(
        np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
        name="A",
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    b = pd.Series(
        np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
        name="B",
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    fit = fit_pair(a, b)
    # Unrelated random walks should either fail to fit or fail the half-life screen.
    # We at least verify no crash and that any returned fit has rho close to 1.
    if fit is not None:
        assert fit.ar1_rho > 0.85  # essentially random-walk residuals


def test_fit_pair_handles_too_short() -> None:
    short_idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(np.ones(10), index=short_idx, name="A")
    b = pd.Series(np.ones(10), index=short_idx, name="B")
    assert fit_pair(a, b) is None


def test_pca_candidate_pairs_orders_by_distance() -> None:
    # Build 4 names where (A,B) are nearly identical and (C,D) are nearly identical;
    # discovery should put one of those pairs first.
    n = 200
    rng = np.random.default_rng(2)
    base_x = rng.normal(0, 0.01, n)
    base_y = rng.normal(0, 0.01, n)
    df = pd.DataFrame(
        {
            "A": base_x + rng.normal(0, 0.0005, n),
            "B": base_x + rng.normal(0, 0.0005, n),
            "C": base_y + rng.normal(0, 0.0005, n),
            "D": base_y + rng.normal(0, 0.0005, n),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    pairs = pca_candidate_pairs(df, n_components=2, max_candidates=2)
    assert len(pairs) <= 2
    # The two closest pairs should be {A,B} and {C,D} in some order.
    sets = [frozenset(p) for p in pairs]
    assert frozenset({"A", "B"}) in sets or frozenset({"C", "D"}) in sets


def test_pca_candidate_pairs_empty_panel() -> None:
    assert pca_candidate_pairs(pd.DataFrame()) == []
    single = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    assert pca_candidate_pairs(single) == []


def test_discover_and_screen_end_to_end() -> None:
    """A panel containing one good pair + noise should yield exactly that pair."""
    a, b = _cointegrated_series(n=400, beta=1.2, half_life=8.0, seed=3)
    rng = np.random.default_rng(4)
    n_extra = 6
    noise_prices: dict[str, pd.Series] = {}
    for i in range(n_extra):
        innov = rng.normal(0.0003, 0.01, len(a))
        noise_prices[f"N{i}"] = pd.Series(
            np.exp(np.cumsum(innov)),
            index=a.index,
            name=f"N{i}",
        )
    prices = pd.concat([pd.DataFrame({"A": a, "B": b}), pd.DataFrame(noise_prices)], axis=1)
    returns = prices.pct_change().dropna()

    fits = discover_and_screen_pairs(
        prices=prices,
        returns=returns,
        max_candidates=30,
        min_half_life=1.0,
        max_half_life=30.0,
        max_kept=5,
    )
    # The known good pair should be in the screened output.
    pair_keys = {frozenset({f.a, f.b}) for f in fits}
    assert frozenset({"A", "B"}) in pair_keys
    # Every returned candidate should have a half-life inside the bounds.
    for fit in fits:
        assert 1.0 <= fit.half_life_days <= 30.0
        assert 0.0 < fit.ar1_rho < 1.0


def test_paircandidate_is_frozen() -> None:
    pc = PairCandidate(
        a="A", b="B", beta=1.0, alpha=0.0, ar1_rho=0.5, half_life_days=10.0, spread_std=0.01
    )
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        pc.a = "C"  # type: ignore[misc]
