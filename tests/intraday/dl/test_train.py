import numpy as np
import pytest

pytest.importorskip("torch")  # skip cleanly where torch is absent

from quant.intraday.dl.config import DLConfig
from quant.intraday.dl.train import TrainOutput, train_model


def _learnable_data(n=400, window=8, seed=0):
    # y is a clean linear function of the window so loss MUST be able to fall.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window))  # noqa: N806
    w = rng.normal(size=window)
    y = X @ w
    return X, y


def test_train_returns_output_and_loss_curve_length():
    X, y = _learnable_data()  # noqa: N806
    cfg = DLConfig(window=8, hidden_size=8, epochs=10, batch_size=32, seed=1)
    out = train_model(X, y, cfg)
    assert isinstance(out, TrainOutput)
    assert len(out.loss_curve) == cfg.epochs
    assert all(c >= 0 for c in out.loss_curve)


def test_loss_decreases():
    X, y = _learnable_data()  # noqa: N806
    cfg = DLConfig(window=8, hidden_size=16, epochs=30, batch_size=32, seed=1)
    out = train_model(X, y, cfg)
    # training works: end loss is clearly below start loss.
    assert out.loss_curve[-1] < out.loss_curve[0] * 0.7


def test_same_seed_same_machine_determinism():
    X, y = _learnable_data()  # noqa: N806
    cfg = DLConfig(window=8, hidden_size=8, epochs=10, batch_size=32, seed=42)
    a = train_model(X, y, cfg)
    b = train_model(X, y, cfg)
    assert a.loss_curve == b.loss_curve  # identical run on the same machine


def test_train_model_restores_global_determinism_flag():
    import torch

    X, y = _learnable_data()  # noqa: N806
    cfg = DLConfig(window=8, hidden_size=8, epochs=3, batch_size=32, seed=1)
    original = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)  # prior state OFF
        train_model(X, y, cfg)
        # train_model leaves no process-wide side-effect: the flag is restored, not left True.
        assert not torch.are_deterministic_algorithms_enabled()

        torch.use_deterministic_algorithms(True)  # prior state ON
        train_model(X, y, cfg)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(original)  # leave the suite's global state clean
