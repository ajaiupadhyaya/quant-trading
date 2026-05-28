from __future__ import annotations

import numpy as np

from quant.regime.kalman_state import kalman_local_level


def test_smoother_reduces_noise_and_is_causal():
    rng = np.random.default_rng(0)
    truth = np.linspace(0.0, 1.0, 200)
    noisy = truth + rng.normal(0, 0.3, size=200)
    smooth = kalman_local_level(noisy, process_var=1e-4, obs_var=1e-1)
    assert smooth.shape == (200,)
    # Smoothed series tracks truth better than the noisy input.
    assert np.mean((smooth - truth) ** 2) < np.mean((noisy - truth) ** 2)
    # Online/causal: value at t unchanged by future observations.
    smooth_trunc = kalman_local_level(noisy[:100], process_var=1e-4, obs_var=1e-1)
    np.testing.assert_allclose(smooth[:100], smooth_trunc, atol=1e-12)


def test_short_input_returns_input_copy():
    assert kalman_local_level(np.array([])).shape == (0,)
