import numpy as np

from quant.intraday.dl.evaluate import (
    oos_metrics,
    random_series,
    synthetic_signal_series,
)


def test_oos_metrics_perfect_prediction():
    y = np.array([1.0, -2.0, 3.0, -0.5])
    m = oos_metrics(y, y.copy())
    assert m["mse"] == 0.0
    assert m["directional_accuracy"] == 1.0
    assert abs(m["r2"] - 1.0) < 1e-12


def test_oos_metrics_directional_accuracy():
    y_true = np.array([1.0, -1.0, 2.0, -3.0])
    y_pred = np.array([0.5, 1.0, 2.0, 4.0])  # signs: +,+,+,+ vs +,-,+,- => 2/4
    m = oos_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == 0.5


def test_random_series_is_reproducible_and_shaped():
    a = random_series(n=500, seed=3)
    b = random_series(n=500, seed=3)
    assert a.shape == (500,)
    assert np.allclose(a, b)
    assert not np.allclose(a, random_series(n=500, seed=4))


def test_synthetic_signal_has_autocorrelation():
    s = synthetic_signal_series(n=4000, seed=1)
    # AR(2) structure => lag-1 autocorrelation clearly non-zero.
    s0, s1 = s[:-1], s[1:]
    corr = np.corrcoef(s0, s1)[0, 1]
    assert abs(corr) > 0.1
