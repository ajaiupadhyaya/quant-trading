import numpy as np

from quant.options.beta import rolling_beta


def test_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 500)
    y = 0.8 * x + rng.normal(0, 1e-5, 500)
    assert abs(rolling_beta(y, x) - 0.8) < 0.02


def test_degenerate_returns_neutral():
    assert rolling_beta(np.array([]), np.array([])) == 1.0
    assert rolling_beta(np.array([0.01]), np.array([0.01])) == 1.0
    assert rolling_beta(np.array([0.01, 0.02]), np.array([0.0, 0.0])) == 1.0  # zero var


def test_clamped_to_range():
    x = np.array([0.001, -0.001, 0.001, -0.001])
    y = 10.0 * x  # beta 10 -> clamp to 3
    assert rolling_beta(y, x) == 3.0
