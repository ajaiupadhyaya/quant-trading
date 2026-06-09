import pytest

from quant.intraday.rl.config import RLConfig


def test_defaults():
    c = RLConfig()
    assert c.total_shares >= 1
    assert c.n_steps >= 1
    assert c.n_actions >= 2
    assert 0 < c.alpha <= 1
    assert 0 <= c.gamma_discount <= 1
    assert c.epsilon_start >= c.epsilon_end >= 0
    assert c.n_episodes >= 1
    assert c.risk_aversion >= 0
    assert c.sigma > 0 and c.dt > 0 and c.start_price > 0
    assert isinstance(c.seed, int)


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        RLConfig(total_shares=0)
    with pytest.raises(ValueError):
        RLConfig(n_actions=1)
    with pytest.raises(ValueError):
        RLConfig(alpha=0.0)
    with pytest.raises(ValueError):
        RLConfig(epsilon_start=0.1, epsilon_end=0.5)
