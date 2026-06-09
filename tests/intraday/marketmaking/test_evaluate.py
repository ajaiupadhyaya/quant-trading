import random

from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.evaluate import SweepPoint, gamma_sweep
from quant.intraday.marketmaking.price_path import abm_path


def test_sweep_returns_point_per_gamma_in_order():
    cfg = MMConfig(horizon_seconds=600.0, dt_seconds=1.0, seed=9)
    prices = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=cfg.n_steps, rng=random.Random(9))
    gammas = [0.01, 0.1, 1.0, 5.0]
    pts = gamma_sweep(prices, cfg, gammas)
    assert [p.gamma for p in pts] == gammas
    assert all(isinstance(p, SweepPoint) for p in pts)


def test_sweep_shows_inventory_control_tradeoff():
    cfg = MMConfig(horizon_seconds=800.0, dt_seconds=1.0, seed=9)
    prices = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=cfg.n_steps, rng=random.Random(9))
    pts = gamma_sweep(prices, cfg, [0.01, 5.0])
    assert pts[1].max_abs_inventory <= pts[0].max_abs_inventory
