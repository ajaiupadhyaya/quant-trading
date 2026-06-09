import numpy as np

from quant.intraday.dl.baselines import linear_predict, naive_predict


def test_naive_predicts_last_in_window():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # noqa: N806
    yhat = naive_predict(X)
    assert list(yhat) == [3.0, 6.0]  # the last in-window value


def test_linear_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(200, 3))  # noqa: N806
    true = np.array([0.5, -0.2, 1.0])
    ytr = Xtr @ true + 0.3  # exact linear + intercept, no noise
    Xte = rng.normal(size=(50, 3))  # noqa: N806
    yhat = linear_predict(Xtr, ytr, Xte)
    expected = Xte @ true + 0.3
    assert np.allclose(yhat, expected, atol=1e-6)


def test_baselines_do_not_import_torch():
    import quant.intraday.dl.baselines as b

    assert not hasattr(b, "torch")
