"""Configuration for the intraday live sleeve. All thresholds live here (no magic
numbers per the Charter); the 'tight & safe' profile is the default."""

from __future__ import annotations

from dataclasses import dataclass

from quant.data.universe import SLEEVE_UNIVERSE


@dataclass(frozen=True)
class SleeveConfig:
    universe: tuple[str, ...] = SLEEVE_UNIVERSE
    notional_cap_pct: float = 0.10        # fraction of paper equity
    notional_cap_abs: float = 10_000.0    # hard $ cap on sleeve notional
    per_trade_cap: float = 2_000.0        # max $ notional per single order
    max_round_trips: int = 20             # max opens per day
    daily_loss_halt_pct: float = 0.015    # of sleeve allocation -> auto-flatten+halt
    flat_by_close_minutes: int = 15       # flatten this many min before close
    tick_seconds: int = 60
    mean_reversion_lookback: int = 30     # ticks for the rolling mean/vol
    entry_z: float = 2.0                  # |z| beyond this -> fade
    exit_z: float = 0.5                   # revert inside this -> exit

    def __post_init__(self) -> None:
        for name in ("notional_cap_pct", "notional_cap_abs", "per_trade_cap"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.max_round_trips <= 0:
            raise ValueError("max_round_trips must be positive")
        if self.entry_z <= self.exit_z:
            raise ValueError(
                f"entry_z must exceed exit_z, got entry_z={self.entry_z} exit_z={self.exit_z}"
            )

    def sleeve_allocation(self, equity: float) -> float:
        """Dollar capital the sleeve may deploy: min(pct of equity, absolute cap)."""
        return min(equity * self.notional_cap_pct, self.notional_cap_abs)
