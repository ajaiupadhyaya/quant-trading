from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.regime.hmm import forward_filter, log_emission
from quant.regime.models import HMMParams


def _toy_params() -> HMMParams:
    return HMMParams(
        start_prob=np.array([0.6, 0.4]),
        trans_mat=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[0.0], [3.0]]),
        variances=np.array([[1.0], [1.0]]),
    )


def test_log_emission_shape_and_peak():
    obs = np.array([[0.0], [3.0]])
    le = log_emission(obs, _toy_params())
    assert le.shape == (2, 2)
    # Obs 0 (value 0) most likely under state 0; obs 1 (value 3) under state 1.
    assert le[0, 0] > le[0, 1]
    assert le[1, 1] > le[1, 0]


def test_forward_filter_is_normalized_and_causal():
    obs = np.array([[0.0], [0.0], [3.0], [3.0]])
    post = forward_filter(obs, _toy_params())
    assert post.shape == (4, 2)
    np.testing.assert_allclose(post.sum(axis=1), np.ones(4), atol=1e-9)
    # Filtered posterior at t depends only on obs[:t+1]: truncating later obs
    # must not change earlier rows.
    post_trunc = forward_filter(obs[:2], _toy_params())
    np.testing.assert_allclose(post[:2], post_trunc, atol=1e-12)


def test_fit_recovers_known_params_and_is_seed_reproducible():
    rng = np.random.default_rng(0)
    true = HMMParams(
        start_prob=np.array([1.0, 0.0]),
        trans_mat=np.array([[0.97, 0.03], [0.05, 0.95]]),
        means=np.array([[0.0], [5.0]]),
        variances=np.array([[0.5], [0.5]]),
    )
    # Generate a long sample from `true`.
    states = np.zeros(1000, dtype=int)
    for t in range(1, states.size):
        states[t] = rng.choice(2, p=true.trans_mat[states[t - 1]])
    obs = true.means[states] + rng.normal(0, np.sqrt(0.5), size=(1000, 1))

    from quant.regime.hmm import fit_hmm

    fit_a = fit_hmm(obs, n_states=2, n_restarts=2, seed=7)
    fit_b = fit_hmm(obs, n_states=2, n_restarts=2, seed=7)

    # Seed reproducibility.
    np.testing.assert_allclose(fit_a.means, fit_b.means)

    # Recover the two cluster means (order-agnostic).
    recovered = np.sort(fit_a.means.ravel())
    np.testing.assert_allclose(recovered, np.array([0.0, 5.0]), atol=0.4)

    # Transition matrix rows are valid distributions.
    np.testing.assert_allclose(fit_a.trans_mat.sum(axis=1), np.ones(2), atol=1e-9)


def test_viterbi_recovers_path_and_score_is_finite():
    from quant.regime.hmm import score, viterbi

    params = HMMParams(
        start_prob=np.array([0.5, 0.5]),
        trans_mat=np.array([[0.95, 0.05], [0.05, 0.95]]),
        means=np.array([[0.0], [10.0]]),
        variances=np.array([[0.25], [0.25]]),
    )
    # Low-noise: first 20 near 0 (state 0), next 20 near 10 (state 1).
    obs = np.concatenate([np.zeros(20), np.full(20, 10.0)])[:, None] + np.random.default_rng(
        1
    ).normal(0, 0.05, size=(40, 1))
    path = viterbi(obs, params)
    assert path.shape == (40,)
    assert path[:20].tolist() == [0] * 20
    assert path[20:].tolist() == [1] * 20
    assert np.isfinite(score(obs, params))


@settings(max_examples=25, deadline=None)
@given(
    n_obs=st.integers(min_value=40, max_value=120),
    seed=st.integers(min_value=0, max_value=50),
)
def test_forward_filter_rows_are_distributions(n_obs: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1, size=(n_obs, 2))
    params = HMMParams(
        start_prob=np.array([0.4, 0.3, 0.3]),
        trans_mat=np.full((3, 3), 1 / 3),
        means=rng.normal(0, 1, size=(3, 2)),
        variances=np.abs(rng.normal(1, 0.2, size=(3, 2))) + 0.1,
    )
    post = forward_filter(obs, params)
    assert np.all(post >= -1e-9)
    np.testing.assert_allclose(post.sum(axis=1), np.ones(n_obs), atol=1e-9)
