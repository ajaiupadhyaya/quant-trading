import numpy as np

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.qlearning import TrainResult, train


def _small() -> RLConfig:
    return RLConfig(total_shares=20, n_steps=10, n_actions=5, n_episodes=4000, seed=3)


def test_qtable_shape_and_determinism() -> None:
    cfg = _small()
    r1 = train(cfg)
    r2 = train(cfg)
    assert isinstance(r1, TrainResult)
    assert r1.qtable.shape == (cfg.n_steps + 1, cfg.total_shares + 1, cfg.n_actions)
    assert np.array_equal(r1.qtable, r2.qtable)
    assert len(r1.training_curve) > 0
    assert np.isfinite(r1.qtable).all()


def test_training_curve_improves() -> None:
    cfg = _small()
    r = train(cfg)
    assert r.training_curve[-1] < r.training_curve[0]
