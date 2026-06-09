import pytest

from quant.intraday.dl.config import DLConfig


def test_defaults():
    c = DLConfig()
    assert c.window >= 1
    assert c.hidden_size >= 1
    assert c.n_layers >= 1
    assert c.lr > 0
    assert c.epochs >= 1
    assert c.batch_size >= 1
    assert 0 < c.train_frac < 1
    assert isinstance(c.seed, int)


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        DLConfig(window=0)
    with pytest.raises(ValueError):
        DLConfig(hidden_size=0)
    with pytest.raises(ValueError):
        DLConfig(n_layers=0)
    with pytest.raises(ValueError):
        DLConfig(lr=0.0)
    with pytest.raises(ValueError):
        DLConfig(epochs=0)
    with pytest.raises(ValueError):
        DLConfig(batch_size=0)
    with pytest.raises(ValueError):
        DLConfig(train_frac=0.0)
    with pytest.raises(ValueError):
        DLConfig(train_frac=1.0)


def test_config_does_not_import_torch():
    import quant.intraday.dl.config as cfg

    assert not hasattr(cfg, "torch")
