import random

from quant.intraday.marketmaking.price_path import abm_path


def test_length_and_start():
    path = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=50, rng=random.Random(1))
    assert len(path) == 51          # s_0 .. s_n
    assert path[0] == 100.0


def test_seed_determinism():
    a = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=50, rng=random.Random(3))
    b = abm_path(s0=100.0, sigma=0.02, dt=1.0, n_steps=50, rng=random.Random(3))
    assert a == b


def test_zero_sigma_is_flat():
    path = abm_path(s0=100.0, sigma=0.0, dt=1.0, n_steps=10, rng=random.Random(1))
    assert all(p == 100.0 for p in path)
