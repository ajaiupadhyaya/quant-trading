"""Tests for the hand-rolled deterministic gradient-boosted regression-tree learner."""

from __future__ import annotations

import numpy as np

from quant.forecast.gbm import GBMConfig, fit_gbm, predict_gbm


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_constant_target_predicts_constant() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    y = np.full(50, 4.2)
    model = fit_gbm(x, y, GBMConfig(n_estimators=20))
    pred = predict_gbm(model, x)
    assert np.allclose(pred, 4.2, atol=1e-6)


def test_beats_mean_baseline_on_nonlinear_signal() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, size=(400, 2))
    # Non-linear, interaction-y target a linear model handles poorly.
    y = np.sin(1.5 * x[:, 0]) + (x[:, 1] ** 2) * 0.3 + rng.normal(0, 0.05, size=400)
    model = fit_gbm(x, y, GBMConfig(n_estimators=150, learning_rate=0.1, max_depth=3))
    pred = predict_gbm(model, x)
    baseline = _rmse(y, np.full_like(y, y.mean()))
    fitted = _rmse(y, pred)
    assert fitted < 0.5 * baseline  # learns the structure


def test_recovers_monotone_single_feature() -> None:
    x = np.linspace(-3, 3, 200).reshape(-1, 1)
    y = (x[:, 0] > 0).astype(float)  # step function — trees should nail it
    model = fit_gbm(x, y, GBMConfig(n_estimators=80, learning_rate=0.2, max_depth=2))
    pred = predict_gbm(model, x)
    assert pred[x[:, 0] > 1.0].mean() > pred[x[:, 0] < -1.0].mean() + 0.7


def test_deterministic_same_inputs_same_output() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 4))
    y = x @ np.array([1.0, -0.5, 0.0, 2.0]) + rng.normal(0, 0.1, size=120)
    cfg = GBMConfig(n_estimators=60, subsample=0.7, seed=7)
    p1 = predict_gbm(fit_gbm(x, y, cfg), x)
    p2 = predict_gbm(fit_gbm(x, y, cfg), x)
    assert np.array_equal(p1, p2)


def test_subsample_seed_changes_fit_but_stays_reasonable() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 3))
    y = x[:, 0] * 1.5 - x[:, 2] + rng.normal(0, 0.1, size=200)
    a = predict_gbm(fit_gbm(x, y, GBMConfig(subsample=0.6, seed=1)), x)
    b = predict_gbm(fit_gbm(x, y, GBMConfig(subsample=0.6, seed=2)), x)
    assert not np.array_equal(a, b)  # different seed -> different subsamples
    assert _rmse(y, a) < _rmse(y, np.full_like(y, y.mean()))


def test_min_samples_leaf_limits_tree_growth() -> None:
    # A huge min_samples_leaf forces shallow/near-constant trees -> ~mean prediction.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 2))
    y = x[:, 0] * 3.0
    model = fit_gbm(x, y, GBMConfig(n_estimators=30, min_samples_leaf=60, max_depth=4))
    pred = predict_gbm(model, x)
    assert _rmse(y, pred) > 0.5 * _rmse(y, np.full_like(y, y.mean()))


def test_predict_shape_and_finiteness() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(40, 5))
    y = rng.normal(size=40)
    pred = predict_gbm(fit_gbm(x, y, GBMConfig(n_estimators=10)), rng.normal(size=(12, 5)))
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))
