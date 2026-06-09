"""Deterministic CPU training loop (Adam + MSE) for the LSTM alpha. torch imported
LAZILY. Reproducible for two runs on the SAME machine (torch does not guarantee bitwise
cross-machine determinism), so tests assert same-run/same-machine determinism only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant.intraday.dl.config import DLConfig


@dataclass
class TrainOutput:
    model: Any  # torch.nn.Module
    loss_curve: list[float]


def train_model(
    x_train: NDArray[np.float64], y_train: NDArray[np.float64], config: DLConfig
) -> TrainOutput:
    """Train the LSTM regressor; return the trained model + per-epoch mean loss."""
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)

    from quant.intraday.dl.model import build_model

    model = build_model(config)
    # (n, window) -> (n, window, 1) for the 1-feature LSTM.
    x_t = torch.tensor(np.asarray(x_train, dtype=np.float32)).unsqueeze(-1)
    y_t = torch.tensor(np.asarray(y_train, dtype=np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    n = x_t.shape[0]

    loss_curve: list[float] = []
    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        loss_curve.append(epoch_loss / n_batches)

    return TrainOutput(model=model, loss_curve=loss_curve)
