import math
import random

from quant.intraday.marketmaking.config import MMConfig
from quant.intraday.marketmaking.price_path import abm_path
from quant.intraday.marketmaking.simulator import MMResult, run_market_making


def _prices(n, seed=11):
    return abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=n, rng=random.Random(seed))


def test_result_shape_and_determinism():
    cfg = MMConfig(horizon_seconds=500.0, dt_seconds=1.0, seed=5)
    prices = _prices(cfg.n_steps)
    r1 = run_market_making(prices, cfg)
    r2 = run_market_making(prices, cfg)
    assert isinstance(r1, MMResult)
    assert r1 == r2
    assert r1.n_bid_fills >= 0 and r1.n_ask_fills >= 0
    assert math.isfinite(r1.final_pnl)
    assert len(r1.inventory_path) == len(prices)


def test_pnl_conservation():
    cfg = MMConfig(horizon_seconds=400.0, dt_seconds=1.0, seed=2)
    prices = _prices(cfg.n_steps)
    r = run_market_making(prices, cfg)
    assert math.isclose(r.final_pnl, r.cash + r.terminal_inventory * prices[-1], rel_tol=1e-9)


def test_higher_gamma_controls_inventory():
    prices = _prices(800)
    cfg_lo = MMConfig(gamma=0.01, horizon_seconds=800.0, dt_seconds=1.0, seed=4)
    cfg_hi = MMConfig(gamma=2.0, horizon_seconds=800.0, dt_seconds=1.0, seed=4)
    lo = run_market_making(prices, cfg_lo)
    hi = run_market_making(prices, cfg_hi)
    assert hi.max_abs_inventory <= lo.max_abs_inventory
