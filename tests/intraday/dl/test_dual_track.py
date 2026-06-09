import pytest

torch = pytest.importorskip("torch")

from quant.intraday.dl.config import DLConfig  # noqa: E402
from quant.intraday.dl.evaluate import (  # noqa: E402
    evaluate_track,
    random_series,
    synthetic_signal_series,
)

# Modest config so the integration test runs in a few seconds, still enough to learn.
_CFG = DLConfig(window=12, hidden_size=24, epochs=40, batch_size=64, seed=7, train_frac=0.7)


def test_synthetic_track_lstm_beats_naive():
    series = synthetic_signal_series(n=3000, seed=7)
    res = evaluate_track(series, _CFG)
    # Machinery works: the LSTM extracts the AR structure -> lower OOS MSE than persistence.
    assert res["lstm"]["mse"] < res["naive"]["mse"]
    # And training actually happened: loss fell.
    assert res["loss_curve"][-1] < res["loss_curve"][0]


def test_random_track_lstm_not_catastrophic():
    series = random_series(n=3000, seed=7)
    res = evaluate_track(series, _CFG)
    # Sanity bound: the LSTM is not catastrophically worse than naive persistence.
    assert res["lstm"]["mse"] <= res["naive"]["mse"] * 1.5


def test_random_track_lstm_has_no_edge_over_linear():
    series = random_series(n=3000, seed=7)
    res = evaluate_track(series, _CFG)
    # The HONEST no-edge claim. On iid-noise returns, naive persistence is a *bad*
    # baseline (it predicts a lagged independent draw, MSE ~ 2*sigma^2), so beating naive
    # is trivial and proves nothing. The informative comparison is LSTM vs the linear
    # baseline: with no learnable structure the LSTM must NOT meaningfully beat OLS. We
    # assert the LSTM does not undercut linear's MSE by more than 5% (no real edge), and
    # that its directional accuracy is near chance (no exploitable sign skill).
    assert res["lstm"]["mse"] >= res["linear"]["mse"] * 0.95
    assert abs(res["lstm"]["directional_accuracy"] - 0.5) < 0.1


def test_random_track_has_no_economic_edge():
    series = random_series(n=3000, seed=7)
    res = evaluate_track(series, _CFG, cost_per_turn=0.02)
    # Honest economics on iid-noise returns: the sign-of-prediction rule shows no real
    # edge (per-bar gross Sharpe is within noise of zero across ~880 test bars), and
    # charging turnover costs only makes the net worse — it never conjures a profit.
    assert abs(res["lstm"]["sharpe_gross"]) < 0.2
    assert res["lstm"]["mean_net"] < res["lstm"]["mean_gross"]
