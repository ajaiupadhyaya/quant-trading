import math

import pytest

from quant.options.pricing import Greeks, bs_greeks, bs_price, implied_vol

S, K, T, VOL, R, Q = 100.0, 100.0, 1.0, 0.20, 0.03, 0.0


def test_atm_call_known_value():
    # Textbook ATM call, S=K=100, vol=20%, r=3%, q=0, T=1y ~= 9.413
    price = bs_price(S, K, T, VOL, R, Q, "call")
    assert price == pytest.approx(9.4134, abs=1e-3)


def test_put_call_parity():
    c = bs_price(S, K, T, VOL, R, Q, "call")
    p = bs_price(S, K, T, VOL, R, Q, "put")
    lhs = c - p
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_delta_matches_finite_difference():
    h = 1e-4
    up = bs_price(S + h, K, T, VOL, R, Q, "call")
    dn = bs_price(S - h, K, T, VOL, R, Q, "call")
    fd_delta = (up - dn) / (2 * h)
    assert bs_greeks(S, K, T, VOL, R, Q, "call").delta == pytest.approx(fd_delta, abs=1e-5)


def test_gamma_matches_finite_difference():
    h = 1e-2
    up = bs_price(S + h, K, T, VOL, R, Q, "call")
    mid = bs_price(S, K, T, VOL, R, Q, "call")
    dn = bs_price(S - h, K, T, VOL, R, Q, "call")
    fd_gamma = (up - 2 * mid + dn) / (h * h)
    assert bs_greeks(S, K, T, VOL, R, Q, "call").gamma == pytest.approx(fd_gamma, abs=1e-4)


def test_vega_matches_finite_difference():
    h = 1e-4
    up = bs_price(S, K, T, VOL + h, R, Q, "call")
    dn = bs_price(S, K, T, VOL - h, R, Q, "call")
    fd_vega = (up - dn) / (2 * h)
    assert bs_greeks(S, K, T, VOL, R, Q, "call").vega == pytest.approx(fd_vega, abs=1e-3)


def test_put_delta_negative():
    assert bs_greeks(S, K, T, VOL, R, Q, "put").delta < 0.0


def test_implied_vol_round_trip():
    price = bs_price(S, 95.0, T, 0.25, R, Q, "put")
    iv = implied_vol(price, S, 95.0, T, R, Q, "put")
    assert iv == pytest.approx(0.25, abs=1e-4)


def test_at_expiry_returns_intrinsic():
    assert bs_price(110.0, 100.0, 0.0, VOL, R, Q, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.0, VOL, R, Q, "call") == pytest.approx(0.0)
    assert bs_price(90.0, 100.0, 0.0, VOL, R, Q, "put") == pytest.approx(10.0)


def test_nonfinite_inputs_return_nan():
    assert math.isnan(bs_price(float("nan"), K, T, VOL, R, Q, "call"))
    assert math.isnan(bs_price(S, K, T, -0.1, R, Q, "call"))


def test_greeks_is_frozen_dataclass():
    g = bs_greeks(S, K, T, VOL, R, Q, "call")
    assert isinstance(g, Greeks)
    with pytest.raises(Exception):
        g.delta = 0.0  # type: ignore[misc]


def test_implied_vol_unreachable_price_is_nan():
    # Price above no-arb upper bound for a call (>= S) -> nan
    assert math.isnan(implied_vol(200.0, S, K, T, R, Q, "call"))
