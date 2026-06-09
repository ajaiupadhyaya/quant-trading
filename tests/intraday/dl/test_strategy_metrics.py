"""Economics of the sign-of-prediction rule. Pure numpy — no torch (evaluate.py is
torch-free at import; torch is only pulled in lazily inside predict())."""

import numpy as np

from quant.intraday.dl.evaluate import strategy_metrics


def test_perfect_directional_prediction_is_profitable():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    m = strategy_metrics(returns, returns.copy(), cost_per_turn=0.0)  # perfect sign
    assert m["hit_rate"] == 1.0
    assert m["mean_gross"] > 0.0
    assert m["sharpe_gross"] > 0.0
    # no cost charged -> net equals gross
    assert m["mean_net"] == m["mean_gross"]


def test_wrong_directional_prediction_loses():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    m = strategy_metrics(returns, -returns, cost_per_turn=0.0)  # always wrong side
    assert m["hit_rate"] == 0.0
    assert m["mean_gross"] < 0.0


def test_costs_only_reduce_net_never_help():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    pred = returns.copy()
    free = strategy_metrics(returns, pred, cost_per_turn=0.0)
    charged = strategy_metrics(returns, pred, cost_per_turn=0.01)
    assert charged["mean_net"] < free["mean_net"]
    assert charged["sharpe_net"] < free["sharpe_net"]
    # gross is unchanged by the cost
    assert charged["mean_gross"] == free["mean_gross"]


def test_flat_predictions_have_zero_pnl_and_turnover():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    m = strategy_metrics(returns, np.zeros_like(returns), cost_per_turn=0.05)  # sign 0 -> flat
    assert m["mean_gross"] == 0.0
    assert m["mean_net"] == 0.0
    assert m["sharpe_gross"] == 0.0
    assert m["avg_turnover"] == 0.0


def test_turnover_counts_position_flips_from_flat():
    # pred signs +,+,-,- -> positions [1,1,-1,-1], starting flat (prev[0]=0).
    returns = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.0, 1.0, -1.0, -1.0])
    m = strategy_metrics(returns, pred, cost_per_turn=1.0)
    # turnover = |1-0| + |1-1| + |-1-1| + |-1+1| = 1 + 0 + 2 + 0 = 3 over 4 bars
    assert abs(m["avg_turnover"] - 3.0 / 4.0) < 1e-12
    # gross = pos*returns = [1,1,-1,-1] -> mean 0; mean_net = mean_gross - cost*avg_turnover
    assert m["mean_gross"] == 0.0
    assert abs(m["mean_net"] - (0.0 - 1.0 * 3.0 / 4.0)) < 1e-12
