"""LSTM regressor for next-bar return. torch is imported LAZILY inside build_model so
importing quant.* elsewhere never pays the torch cost. Deterministic init from config.seed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant.intraday.dl.config import DLConfig

if TYPE_CHECKING:  # for type-checkers only; no runtime torch import
    import torch


def build_model(config: DLConfig) -> Any:
    """Construct an LSTMRegressor (1-input LSTM -> linear head -> scalar), seeded for
    deterministic initial weights. Returns a torch.nn.Module."""
    import torch
    from torch import nn

    torch.manual_seed(config.seed)

    class LSTMRegressor(nn.Module):
        def __init__(self, hidden_size: int, n_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, window, 1) -> use the last timestep's hidden state.
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            result: torch.Tensor = self.head(last).squeeze(-1)
            return result

    return LSTMRegressor(config.hidden_size, config.n_layers)
