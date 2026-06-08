"""Almgren-Chriss optimal execution (Almgren & Chriss, 2000, "Optimal execution of
portfolio transactions"). Linear permanent (g(v)=gamma*v) and temporary (h(v)=eta*v)
impact; mean-variance objective E[C] + lambda*V[C].

The optimal holdings follow x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T), where
kappa solves cosh(kappa*tau) = 1 + lambda*sigma^2*tau^2 / (2*eta_tilde),
eta_tilde = eta - gamma*tau/2. We compute the trajectory in closed form, then derive
E[C] and V[C] directly from it (avoids transcribing the messy closed-form cost)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ACPlan:
    child_sizes: list[int]      # n_1..n_N shares per interval (sum = X)
    holdings: list[float]       # x_0..x_N remaining shares (x_0 = X, x_N = 0)
    expected_cost: float
    variance: float
    kappa: float


@dataclass(frozen=True)
class FrontierPoint:
    risk_aversion: float
    expected_cost: float
    variance: float


def _solve_kappa(sigma: float, eta: float, gamma: float, tau: float, lam: float) -> float:
    eta_tilde = eta - gamma * tau / 2.0
    if eta_tilde <= 0:
        eta_tilde = eta
    arg = 1.0 + (lam * sigma * sigma * tau * tau) / (2.0 * eta_tilde)
    if arg <= 1.0:
        return 0.0
    return math.acosh(arg) / tau


def _holdings(total_shares: int, n_intervals: int, tau: float, kappa: float) -> list[float]:
    # Use conventional math notation: x_total (X), horizon (t_horizon), sinh denominator
    n = n_intervals
    x_total = float(total_shares)
    t_horizon = n * tau
    if kappa <= 0.0:
        return [x_total * (1.0 - j / n) for j in range(n + 1)]
    sinh_denom = math.sinh(kappa * t_horizon)
    return [x_total * math.sinh(kappa * (t_horizon - j * tau)) / sinh_denom
            for j in range(n + 1)]


def _child_sizes(holdings: list[float]) -> list[int]:
    raw = [holdings[j - 1] - holdings[j] for j in range(1, len(holdings))]
    sizes = [round(r) for r in raw]
    total = round(holdings[0])
    sizes[-1] += total - sum(sizes)
    return sizes


def optimal_schedule(
    *, total_shares: int, n_intervals: int, tau: float,
    sigma: float, eta: float, gamma: float, risk_aversion: float,
) -> ACPlan:
    kappa = _solve_kappa(sigma, eta, gamma, tau, risk_aversion)
    holdings = _holdings(total_shares, n_intervals, tau, kappa)
    sizes = _child_sizes(holdings)
    expected_cost = 0.5 * gamma * total_shares**2 + (eta / tau) * sum(n * n for n in sizes)
    variance = sigma * sigma * tau * sum(x * x for x in holdings[1:])
    return ACPlan(child_sizes=sizes, holdings=holdings,
                  expected_cost=expected_cost, variance=variance, kappa=kappa)


def efficient_frontier(
    *, total_shares: int, n_intervals: int, tau: float,
    sigma: float, eta: float, gamma: float, lambdas: list[float],
) -> list[FrontierPoint]:
    pts: list[FrontierPoint] = []
    for lam in sorted(lambdas):
        plan = optimal_schedule(total_shares=total_shares, n_intervals=n_intervals,
                                tau=tau, sigma=sigma, eta=eta, gamma=gamma,
                                risk_aversion=lam)
        pts.append(FrontierPoint(risk_aversion=lam,
                                 expected_cost=plan.expected_cost, variance=plan.variance))
    return pts
