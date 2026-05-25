"""Strategy ABC + StrategySpec dataclass.

Concrete strategies land in Plans 4 and 5. Foundation only needs the contract.
Plan 2 adds parameter support so walk-forward can grid-search.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

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

    spec: ClassVar[StrategySpec]
    default_params: ClassVar[dict[str, Any]] = {}
    # Walk-forward parameter grid. Keys are param names; values are lists of
    # candidates to grid-search per training window. An empty dict means
    # "only the defaults" — walk-forward will run once per window with no sweep.
    param_grid: ClassVar[dict[str, list[Any]]] = {}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        # Deep-copy so nested defaults (e.g. {"thresholds": {"entry": 1.0}}) can't
        # be mutated through `self.params` back into the class-level default.
        merged: dict[str, Any] = copy.deepcopy(self.default_params)
        if params:
            merged.update(params)
        self.params: dict[str, Any] = merged

    @abstractmethod
    def generate_signals(self, asof: date) -> pd.Series:
        """Return a Series indexed by symbol with the signal score for each name."""

    @abstractmethod
    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        """Return target whole-share positions keyed by symbol.

        Positive = long, negative = short, missing/zero = no position.
        """

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, Any] | None = None) -> Strategy:
        """Construct a strategy instance, given the bars frame and params.

        Default: ignore bars and instantiate with params only. Strategies that
        need bars at construction (e.g. for caching signal series) override.
        """
        del bars  # unused at the base class
        return cls(params=params)
