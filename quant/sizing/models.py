"""Configuration and decision records for the sizing engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

DEFAULT_REGIME_WEIGHTS: dict[str, float] = {"calm-bull": 1.0, "choppy": 0.5, "crisis": 0.0}


def _default_regime_weights() -> Mapping[str, float]:
    return MappingProxyType(dict(DEFAULT_REGIME_WEIGHTS))


@dataclass(frozen=True)
class SizingConfig:
    """Knobs for the four-component gross-exposure scalar. All defaults intentional."""

    target_vol: float = 0.15
    vol_lookback_days: int = 63
    max_leverage: float = 2.0
    use_vol_target: bool = True

    kelly_fraction: float = 0.5
    kelly_cap: float = 1.0
    kelly_lookback_days: int = 252
    use_kelly: bool = True

    dd_floor: float = 0.20
    dd_lookback_days: int = 252
    use_drawdown: bool = True

    regime_weights: Mapping[str, float] = field(default_factory=_default_regime_weights)
    use_regime: bool = True


@dataclass(frozen=True)
class SizingDecision:
    """A single day's gross scalar plus its post-toggle component breakdown."""

    gross: float
    vol_scale: float
    kelly: float
    drawdown: float
    regime: float
