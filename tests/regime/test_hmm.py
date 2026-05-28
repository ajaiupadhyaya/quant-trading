from __future__ import annotations

import numpy as np

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
