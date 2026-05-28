import numpy as np

from quant.options.models import HedgeConfig
from quant.options.policy import build_hedge


def _hist(n=70, seed=0):
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0003, 0.01, n)
    book = 1.0 * spy + rng.normal(0, 1e-4, n)
    return book, spy


def test_crisis_buys_more_contracts_than_calm():
    book, spy = _hist()
    cfg = HedgeConfig()
    calm = build_hedge(100.0, book, spy, "calm-bull", cfg, 1.0, expiry_index=21)
    crisis = build_hedge(100.0, book, spy, "crisis", cfg, 1.0, expiry_index=21)
    assert crisis.contracts > calm.contracts
    assert crisis.intensity == 1.0
    assert calm.intensity == 0.25


def test_no_regime_full_intensity():
    book, spy = _hist()
    cfg = HedgeConfig(use_regime=False)
    dec = build_hedge(100.0, book, spy, "crisis", cfg, 1.0, expiry_index=21)
    assert dec.intensity == 1.0


def test_unknown_label_neutral_intensity():
    book, spy = _hist()
    cfg = HedgeConfig()
    dec = build_hedge(100.0, book, spy, "mystery", cfg, 1.0, expiry_index=21)
    assert dec.intensity == 1.0


def test_premium_positive_for_put():
    book, spy = _hist()
    cfg = HedgeConfig(structure="put")
    dec = build_hedge(100.0, book, spy, "crisis", cfg, 1.0, expiry_index=21)
    assert dec.premium > 0.0
    assert dec.contracts > 0.0


def test_contracts_scale_with_book_value():
    book, spy = _hist()
    cfg = HedgeConfig()
    small = build_hedge(100.0, book, spy, "choppy", cfg, 1.0, expiry_index=21)
    large = build_hedge(100.0, book, spy, "choppy", cfg, 2.0, expiry_index=21)
    assert large.contracts == 2 * small.contracts
