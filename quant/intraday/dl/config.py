"""Configuration for the intraday DL alpha (LSTM) showcase. No magic numbers per the
Charter; all knobs live here. NO torch import (kept lazy in model/train)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DLConfig:
    window: int = 16
    hidden_size: int = 32
    n_layers: int = 1
    lr: float = 0.01
    epochs: int = 60
    batch_size: int = 64
    seed: int = 7
    train_frac: float = 0.7

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
