import math

from quant.intraday.execution.almgren_chriss import (
    ACPlan,
    efficient_frontier,
    optimal_schedule,
)


def _params(lam):
    return dict(total_shares=1000, n_intervals=10, tau=1.0,
                sigma=0.02, eta=1e-4, gamma=1e-5, risk_aversion=lam)


def test_child_sizes_sum_to_parent():
    plan = optimal_schedule(**_params(1e-6))
    assert sum(plan.child_sizes) == 1000
    assert len(plan.child_sizes) == 10
    assert all(n >= 0 for n in plan.child_sizes)


def test_low_risk_aversion_is_approximately_uniform():
    plan = optimal_schedule(**_params(1e-12))
    sizes = plan.child_sizes
    assert max(sizes) - min(sizes) <= 2  # near-uniform (TWAP-like)


def test_high_risk_aversion_is_front_loaded():
    plan = optimal_schedule(**_params(1e-2))
    assert plan.child_sizes[0] > plan.child_sizes[-1]


def test_cost_and_variance_are_finite_and_positive():
    plan = optimal_schedule(**_params(1e-6))
    assert plan.expected_cost > 0 and math.isfinite(plan.expected_cost)
    assert plan.variance >= 0 and math.isfinite(plan.variance)


def test_efficient_frontier_is_monotone():
    pts = efficient_frontier(total_shares=1000, n_intervals=10, tau=1.0,
                             sigma=0.02, eta=1e-4, gamma=1e-5,
                             lambdas=[1e-8, 1e-6, 1e-4, 1e-2])
    costs = [p.expected_cost for p in pts]
    variances = [p.variance for p in pts]
    assert costs == sorted(costs)
    assert variances == sorted(variances, reverse=True)
