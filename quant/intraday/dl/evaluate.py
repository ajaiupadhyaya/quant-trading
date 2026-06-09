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


def strategy_metrics(
    returns: NDArray[np.float64],
    pred: NDArray[np.float64],
    cost_per_turn: float = 0.0,
) -> dict[str, float]:
    """Economics of a sign-of-prediction long/short rule on realized RAW returns.

    Each bar the position is the sign of the predicted next return (+1 long, -1 short, 0
    flat when the prediction is exactly 0); per-bar P&L is ``position * realized return``,
    minus ``cost_per_turn`` for every unit of position change (turnover, starting flat).
    Sharpe is per-bar (mean/std of P&L, unannualized) — intraday-bar annualization is
    deliberately omitted to avoid a misleading factor. On a near-random return series the
    net Sharpe sits around zero and turns negative once costs bite — the honest result."""
    r = np.asarray(returns, dtype=np.float64)
    pos = np.sign(np.asarray(pred, dtype=np.float64))
    gross = pos * r
    prev = np.concatenate(([0.0], pos[:-1]))  # start flat (no position before the first bar)
    turnover = np.abs(pos - prev)
    net = gross - cost_per_turn * turnover

    def _sharpe(x: NDArray[np.float64]) -> float:
        sd = float(np.std(x))
        return float(np.mean(x) / sd) if sd > 0.0 else 0.0

    return {
        "mean_gross": float(np.mean(gross)),
        "mean_net": float(np.mean(net)),
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "hit_rate": float(np.mean(gross > 0.0)),
        "avg_turnover": float(np.mean(turnover)),
        "cost_per_turn": float(cost_per_turn),
    }


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


def evaluate_track(
    series: NDArray[np.float64], config: DLConfig, cost_per_turn: float = 0.0
) -> dict[str, Any]:
    """Window -> chronological split -> train-only standardize -> compare LSTM vs linear
    vs naive OOS. All three predict in the same standardized space (one train-X (mu, sd)).

    Each model entry carries both statistical metrics (mse/dir-acc/r2, in standardized
    space) and the economics of a sign-of-prediction rule (Sharpe/PnL on the RAW realized
    returns, from de-standardized predictions); ``cost_per_turn`` charges turnover."""
    x, y = make_windows(series, config.window)
    x_tr, y_tr, x_te, y_te = train_test_split(x, y, config.train_frac)
    x_tr_z, x_te_z, mu, sd = standardize(x_tr, x_te)
    y_tr_z = (y_tr - mu) / sd
    y_te_z = (y_te - mu) / sd

    naive_hat = naive_predict(x_te_z)
    linear_hat = linear_predict(x_tr_z, y_tr_z, x_te_z)
    out = train_model(x_tr_z, y_tr_z, config)
    lstm_hat = predict(out.model, x_te_z)

    def _scored(hat_z: NDArray[np.float64]) -> dict[str, float]:
        # Statistical metrics in standardized space; economics on raw returns (y_te) using
        # the sign of the de-standardized prediction (hat_z * sd + mu).
        stats = oos_metrics(y_te_z, hat_z)
        econ = strategy_metrics(y_te, hat_z * sd + mu, cost_per_turn)
        return {**stats, **econ}

    return {
        "naive": _scored(naive_hat),
        "linear": _scored(linear_hat),
        "lstm": _scored(lstm_hat),
        "loss_curve": out.loss_curve,
    }
