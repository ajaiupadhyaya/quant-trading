import numpy as np

from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.policy import greedy_action, rollout_schedule
from quant.intraday.rl.qlearning import train


def test_greedy_action_picks_argmax():
    q = np.zeros((3, 5, 4))
    q[1, 2, 3] = 5.0
    assert greedy_action(q, (1, 2)) == 3


def test_rollout_schedule_liquidates_full_parent():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5, n_episodes=4000, seed=3)
    r = train(cfg)
    sched = rollout_schedule(r.qtable, cfg)
    assert sum(sched) == cfg.total_shares
    assert all(n >= 0 for n in sched)
    assert len(sched) == cfg.n_steps
