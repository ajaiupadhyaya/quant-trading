import pytest

from quant.intraday.marketmaking.config import MMConfig


def test_defaults():
    c = MMConfig()
    assert c.gamma > 0
    assert c.k > 0
    assert c.fill_rate_a > 0
    assert c.horizon_seconds > 0
    assert c.dt_seconds > 0
    assert c.sigma > 0
    assert c.lot_size >= 1
    assert isinstance(c.seed, int)


def test_n_steps_is_horizon_over_dt():
    c = MMConfig(horizon_seconds=100.0, dt_seconds=2.0)
    assert c.n_steps == 50


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        MMConfig(gamma=0.0)
    with pytest.raises(ValueError):
        MMConfig(k=0.0)
    with pytest.raises(ValueError):
        MMConfig(dt_seconds=0.0)
    with pytest.raises(ValueError):
        MMConfig(lot_size=0)
