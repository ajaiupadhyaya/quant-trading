import math

from quant.intraday.marketmaking.avellaneda_stoikov import (
    optimal_spread,
    quotes,
    reservation_price,
)


def test_reservation_price_skews_with_inventory():
    mid, gamma, sigma, tau = 100.0, 0.1, 0.02, 300.0
    assert reservation_price(mid, 0, gamma, sigma, tau) == mid
    assert reservation_price(mid, 5, gamma, sigma, tau) < mid
    assert reservation_price(mid, -5, gamma, sigma, tau) > mid


def test_optimal_spread_properties():
    """Spread is positive and increases with sigma and t_remaining.

    Note: the A-S spread is NOT monotone in gamma — the (2/gamma)*ln(1+gamma/k)
    term decreases faster than gamma*sigma^2*tau rises for typical params, so
    we do not assert monotonicity in gamma.  We verify the exact closed-form
    value instead to confirm the formula is implemented correctly.
    """
    base = optimal_spread(gamma=0.1, sigma=0.02, t_remaining=300.0, k=1.5)
    # Positivity
    assert base > 0
    # Monotone in sigma (sigma enters as sigma^2 in both terms)
    assert optimal_spread(gamma=0.1, sigma=0.04, t_remaining=300.0, k=1.5) > base
    # Monotone in t_remaining
    assert optimal_spread(gamma=0.1, sigma=0.02, t_remaining=600.0, k=1.5) > base
    # Exact formula: gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k)
    expected = 0.1 * 0.02**2 * 300.0 + (2.0 / 0.1) * math.log(1.0 + 0.1 / 1.5)
    assert math.isclose(base, expected, rel_tol=1e-12)


def test_quotes_symmetric_about_reservation_price():
    mid, q, gamma, sigma, tau, k = 100.0, 3, 0.1, 0.02, 300.0, 1.5
    bid, ask = quotes(mid, q, gamma, sigma, tau, k)
    r = reservation_price(mid, q, gamma, sigma, tau)
    spread = optimal_spread(gamma=gamma, sigma=sigma, t_remaining=tau, k=k)
    assert math.isclose((bid + ask) / 2.0, r, rel_tol=1e-9)
    assert math.isclose(ask - bid, spread, rel_tol=1e-9)
    assert bid < ask


def test_long_inventory_pushes_both_quotes_down_vs_flat():
    flat_bid, flat_ask = quotes(100.0, 0, 0.1, 0.02, 300.0, 1.5)
    long_bid, long_ask = quotes(100.0, 5, 0.1, 0.02, 300.0, 1.5)
    assert long_bid < flat_bid and long_ask < flat_ask
