import random

from quant.intraday.marketmaking.intensity import (
    draws_fill,
    fill_intensity,
    fill_probability,
)


def test_intensity_decays_with_distance():
    near = fill_intensity(delta=0.1, a=140.0, k=1.5)
    far = fill_intensity(delta=1.0, a=140.0, k=1.5)
    assert near > far > 0.0


def test_probability_in_unit_interval_and_monotone():
    p_near = fill_probability(delta=0.05, a=140.0, k=1.5, dt=1.0)
    p_far = fill_probability(delta=2.0, a=140.0, k=1.5, dt=1.0)
    assert 0.0 <= p_far <= p_near <= 1.0


def test_probability_far_quote_approaches_zero():
    assert fill_probability(delta=100.0, a=140.0, k=1.5, dt=1.0) < 1e-6


def test_negative_distance_clamps_to_one():
    p = fill_probability(delta=-1.0, a=140.0, k=1.5, dt=1.0)
    assert 0.0 <= p <= 1.0


def test_draws_fill_is_seed_deterministic():
    r1, r2 = random.Random(1), random.Random(1)
    seq1 = [draws_fill(0.5, r1) for _ in range(20)]
    seq2 = [draws_fill(0.5, r2) for _ in range(20)]
    assert seq1 == seq2
    assert any(seq1) and not all(seq1)
