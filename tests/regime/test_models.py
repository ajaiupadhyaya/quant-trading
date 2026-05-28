from __future__ import annotations

import numpy as np

from quant.regime.models import N_STATES, REGIME_LABELS, HMMParams


def test_regime_label_constants():
    assert REGIME_LABELS == ("calm-bull", "choppy", "crisis")
    assert N_STATES == 3


def test_hmmparams_shapes_and_roundtrip():
    params = HMMParams(
        start_prob=np.array([0.5, 0.3, 0.2]),
        trans_mat=np.full((3, 3), 1 / 3),
        means=np.zeros((3, 2)),
        variances=np.ones((3, 2)),
    )
    assert params.n_states == 3
    assert params.n_features == 2

    restored = HMMParams.from_json_dict(params.to_json_dict())
    np.testing.assert_allclose(restored.start_prob, params.start_prob)
    np.testing.assert_allclose(restored.trans_mat, params.trans_mat)
    np.testing.assert_allclose(restored.means, params.means)
    np.testing.assert_allclose(restored.variances, params.variances)
