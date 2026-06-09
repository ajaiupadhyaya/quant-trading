import numpy as np
import pytest

from quant.intraday.dl.data import make_windows, standardize, train_test_split


def test_make_windows_shapes_and_contents():
    series = np.arange(10.0)  # 0,1,...,9
    X, y = make_windows(series, window=3)  # noqa: N806
    # n = len - window = 7
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    # X[0] = [0,1,2], y[0] = 3 (next value)
    assert list(X[0]) == [0.0, 1.0, 2.0]
    assert y[0] == 3.0
    # X[-1] = [6,7,8], y[-1] = 9
    assert list(X[-1]) == [6.0, 7.0, 8.0]
    assert y[-1] == 9.0


def test_make_windows_rejects_too_short():
    with pytest.raises(ValueError):
        make_windows(np.arange(3.0), window=3)  # need len > window


def test_train_test_split_is_chronological():
    X = np.arange(20.0).reshape(10, 2)  # noqa: N806
    y = np.arange(10.0)
    Xtr, ytr, Xte, yte = train_test_split(X, y, train_frac=0.7)  # noqa: N806
    assert len(Xtr) == 7 and len(Xte) == 3
    # chronological: train is the FIRST 7, test the LAST 3 (no shuffle)
    assert ytr[0] == 0.0 and ytr[-1] == 6.0
    assert yte[0] == 7.0 and yte[-1] == 9.0


def test_standardize_uses_train_stats_only():
    train = np.array([[0.0, 2.0], [4.0, 6.0]])  # mean 3.0, std sqrt(5)
    test = np.array([[8.0, 10.0]])
    tr_z, te_z, mu, sd = standardize(train, test)
    assert mu == 3.0
    assert abs(sd - np.std(train)) < 1e-12
    # test standardized with TRAIN stats, not its own
    assert np.allclose(te_z, (test - mu) / sd)
    # train standardized to ~zero mean
    assert abs(tr_z.mean()) < 1e-12


def test_standardize_handles_zero_std():
    train = np.ones((4, 2))
    test = np.ones((2, 2))
    tr_z, _te_z, _mu, sd = standardize(train, test)
    assert sd == 1.0  # guarded, no divide-by-zero
    assert np.allclose(tr_z, 0.0)


def test_data_does_not_import_torch():
    import quant.intraday.dl.data as d

    assert not hasattr(d, "torch")
