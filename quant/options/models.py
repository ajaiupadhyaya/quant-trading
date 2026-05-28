"""Dataclasses for the hedging overlay: legs, structures, config, decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant.options.pricing import bs_price

DEFAULT_REGIME_INTENSITY: dict[str, float] = {"calm-bull": 0.25, "choppy": 0.6, "crisis": 1.0}


def _default_regime_intensity() -> Mapping[str, float]:
    return MappingProxyType(dict(DEFAULT_REGIME_INTENSITY))


@dataclass(frozen=True)
class OptionLeg:
    """One leg: right in {"call","put"}, strike, signed quantity (+long/-short)."""

    right: str
    strike: float
    quantity: float


@dataclass(frozen=True)
class HedgeStructure:
    """A built multi-leg structure struck against ``spot_at_open``."""

    legs: tuple[OptionLeg, ...]
    spot_at_open: float
    expiry_index: int

    def value(self, spot: float, t_years: float, vol: float, r: float, q: float) -> float:
        """Sum of signed-leg Black-Scholes values at the given market state."""
        total = 0.0
        for leg in self.legs:
            total += leg.quantity * bs_price(spot, leg.strike, t_years, vol, r, q, leg.right)
        return total


@dataclass(frozen=True)
class HedgeConfig:
    """Knobs for the hedging overlay. All defaults intentional."""

    structure: str = "put"  # "put" | "collar" | "put_spread"
    put_moneyness: float = 0.05
    call_moneyness: float = 0.05
    spread_width: float = 0.10
    coverage: float = 1.0
    tenor_days: int = 30
    roll_days: int = 21
    vol_lookback_days: int = 21
    risk_free: float = 0.03
    div_yield: float = 0.015
    beta_lookback_days: int = 63
    use_regime: bool = True
    regime_intensity: Mapping[str, float] = field(default_factory=_default_regime_intensity)


@dataclass(frozen=True)
class HedgeDecision:
    """A single roll's record for introspection/serialization."""

    structure: HedgeStructure
    contracts: float
    premium: float
    net_beta: float
    regime_label: str | None
    intensity: float
