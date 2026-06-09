"""Out-of-sample evaluation for the DL alpha. Metrics, prediction, seeded series
generators, and the per-track LSTM-vs-baselines comparison. torch is imported LAZILY
inside predict() only; the metrics + generators are pure numpy."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant.intraday.dl.baselines import linear_predict, naive_predict
from quant.intraday.dl.config import DLConfig
from quant.intraday.dl.data import make_windows, standardize, train_test_split
from quant.intraday.dl.train import train_model


def predict(model: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Run the trained LSTM forward (eval mode, no grad). Lazy torch import."""
    import torch

    model.eval()
    x_t = torch.tensor(np.asarray(x, dtype=np.float32)).unsqueeze(-1)
    with torch.no_grad():
        out = model(x_t)
    result: NDArray[np.float64] = out.numpy().astype(np.float64)
    return result


def oos_metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> dict[str, float]:
    """MSE, directional accuracy (sign match), and R^2. Pure numpy."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((yt - yp) ** 2))
    directional_accuracy = float(np.mean(np.sign(yp) == np.sign(yt)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"mse": mse, "directional_accuracy": directional_accuracy, "r2": r2}


def random_series(n: int, seed: int, sigma: float = 1.0) -> NDArray[np.float64]:
    """A near-martingale iid-noise return series (no learnable structure)."""
    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.normal(0.0, sigma, size=n)
    return result


def synthetic_signal_series(
    n: int, seed: int, a: float = 0.6, b: float = -0.3, noise: float = 0.3
) -> NDArray[np.float64]:
    """A stationary AR(2) series r_t = a*r_{t-1} + b*r_{t-2} + eps with KNOWN learnable
    structure. The LSTM MUST beat naive on this track."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, noise, size=n)
    r = np.zeros(n, dtype=np.float64)
    for t in range(2, n):
        r[t] = a * r[t - 1] + b * r[t - 2] + eps[t]
    return r


def evaluate_track(series: NDArray[np.float64], config: DLConfig) -> dict[str, Any]:
    """Window -> chronological split -> train-only standardize -> compare LSTM vs linear
    vs naive OOS. All three predict in the same standardized space (one train-X (mu, sd))."""
    x, y = make_windows(series, config.window)
    x_tr, y_tr, x_te, y_te = train_test_split(x, y, config.train_frac)
    x_tr_z, x_te_z, mu, sd = standardize(x_tr, x_te)
    y_tr_z = (y_tr - mu) / sd
    y_te_z = (y_te - mu) / sd

    naive_hat = naive_predict(x_te_z)
    linear_hat = linear_predict(x_tr_z, y_tr_z, x_te_z)
    out = train_model(x_tr_z, y_tr_z, config)
    lstm_hat = predict(out.model, x_te_z)

    return {
        "naive": oos_metrics(y_te_z, naive_hat),
        "linear": oos_metrics(y_te_z, linear_hat),
        "lstm": oos_metrics(y_te_z, lstm_hat),
        "loss_curve": out.loss_curve,
    }
