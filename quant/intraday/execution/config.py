"""Configuration for the intraday optimal-execution engine. No magic numbers per
the Charter; all knobs live here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecConfig:
    horizon_ticks: int = 5           # number of 60s ticks to work a parent over
    risk_aversion: float = 1e-6      # Almgren-Chriss lambda (per-share-var units)
    perm_impact_frac: float = 0.1    # gamma = perm_impact_frac * eta
    sigma_lookback_bars: int = 60    # bars for realized-vol estimate
    adv_window_bars: int = 20        # bars for trailing dollar-ADV
    impact_coef_bps: float = 10.0    # sqrt-impact coefficient at 100% ADV (bps)

    def __post_init__(self) -> None:
        if self.horizon_ticks <= 0:
            raise ValueError("horizon_ticks must be positive")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if not 0.0 <= self.perm_impact_frac <= 1.0:
            raise ValueError("perm_impact_frac must be in [0, 1]")
        if self.sigma_lookback_bars <= 0 or self.adv_window_bars <= 0:
            raise ValueError("lookback windows must be positive")
        if self.impact_coef_bps <= 0:
            raise ValueError("impact_coef_bps must be positive")
