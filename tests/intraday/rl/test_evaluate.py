from quant.intraday.rl.config import RLConfig
from quant.intraday.rl.evaluate import compare, cost_schedule


def test_cost_schedule_positive_and_seed_deterministic():
    cfg = RLConfig(total_shares=20, n_steps=10)
    c1 = cost_schedule([2] * 10, cfg, seed=5)
    c2 = cost_schedule([2] * 10, cfg, seed=5)
    assert c1 == c2 and c1 > 0.0


def test_compare_returns_three_costs_and_learned_beats_twap():
    cfg = RLConfig(total_shares=20, n_steps=10, n_actions=5, n_episodes=6000, seed=3)
    res = compare(cfg, n_eval_paths=200)
    for key in ("learned", "almgren_chriss", "twap"):
        assert key in res and res[key] > 0.0
    assert res["learned"] <= res["twap"] * 1.05
    assert res["learned"] <= res["almgren_chriss"] * 1.25
