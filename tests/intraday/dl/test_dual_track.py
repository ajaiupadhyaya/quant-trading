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
    # Honest no-edge: on near-random returns the LSTM does NOT beat the baselines, but it
    # must not be catastrophically worse than the naive baseline (within a tolerance band).
    assert res["lstm"]["mse"] <= res["naive"]["mse"] * 1.5
