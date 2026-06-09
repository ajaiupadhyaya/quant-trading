from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.env import ExecutionEnv, step_cost


def test_step_cost_is_nonnegative_and_zero_when_flat_and_no_trade():
    cfg = RLConfig()
    assert step_cost(traded=0, inventory_after=0, price=100.0, config=cfg) == 0.0
    assert step_cost(traded=10, inventory_after=0, price=100.0, config=cfg) > 0.0
    assert step_cost(traded=0, inventory_after=10, price=100.0, config=cfg) > 0.0


def test_reset_returns_full_inventory_and_full_horizon():
    cfg = RLConfig(total_shares=20, n_steps=10)
    env = ExecutionEnv(cfg, seed=1)
    s = env.reset()
    assert s == (10, 20)


def test_step_sells_fraction_and_decrements_state():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5)
    env = ExecutionEnv(cfg, seed=1)
    env.reset()
    (sr, inv), reward, done = env.step(2)
    assert sr == 9 and inv == 10
    assert reward <= 0.0
    assert not done


def test_inventory_never_negative_and_action_clamps():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5)
    env = ExecutionEnv(cfg, seed=1)
    env.reset()
    env.step(4)
    (_sr, inv), _reward, _done = env.step(4)
    assert inv == 0


def test_terminal_force_liquidates_remaining():
    cfg = RLConfig(total_shares=20, n_steps=2, n_actions=5)
    env = ExecutionEnv(cfg, seed=1)
    env.reset()
    env.step(0)
    (sr, inv), reward, done = env.step(0)
    assert done and sr == 0
    assert inv == 0
    assert reward < 0.0
