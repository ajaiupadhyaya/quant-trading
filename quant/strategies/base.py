"""Strategy ABC + StrategySpec dataclass.

Concrete strategies land in Plans 4 and 5. Foundation only needs the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """Static metadata about a strategy."""

    slug: str
    name: str
    description: str
    universe: list[str]
    rebalance_frequency: str  # "daily" | "weekly" | "monthly"
    enabled_live: bool = field(default=False)


class Strategy(ABC):
    """Base class for all strategies. Concrete strategies subclass and register."""

    spec: ClassVar[StrategySpec]  # class attribute provided by subclass

    @abstractmethod
    def generate_signals(self, asof: date) -> pd.Series:
        """Return a Series indexed by symbol with the signal score for each name."""

    @abstractmethod
    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        """Return target whole-share positions keyed by symbol.

        Positive = long, negative = short, missing/zero = no position.
        """
