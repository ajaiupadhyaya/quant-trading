"""Pure builders mapping (spot, config) -> a HedgeStructure (quantity = 1 unit)."""

from __future__ import annotations

from collections.abc import Callable

from quant.options.models import HedgeConfig, HedgeStructure, OptionLeg

# expiry_index is filled by the overlay at roll time; builders use a placeholder.
_PLACEHOLDER_EXPIRY = 0


def protective_put(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    strike = spot * (1.0 - cfg.put_moneyness)
    return HedgeStructure(
        legs=(OptionLeg("put", strike, 1.0),),
        spot_at_open=spot,
        expiry_index=_PLACEHOLDER_EXPIRY,
    )


def collar(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    put_strike = spot * (1.0 - cfg.put_moneyness)
    call_strike = spot * (1.0 + cfg.call_moneyness)
    return HedgeStructure(
        legs=(OptionLeg("put", put_strike, 1.0), OptionLeg("call", call_strike, -1.0)),
        spot_at_open=spot,
        expiry_index=_PLACEHOLDER_EXPIRY,
    )


def put_spread(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    high_strike = spot * (1.0 - cfg.put_moneyness)
    low_strike = spot * (1.0 - cfg.put_moneyness - cfg.spread_width)
    return HedgeStructure(
        legs=(OptionLeg("put", high_strike, 1.0), OptionLeg("put", low_strike, -1.0)),
        spot_at_open=spot,
        expiry_index=_PLACEHOLDER_EXPIRY,
    )


_BUILDERS: dict[str, Callable[[float, HedgeConfig], HedgeStructure]] = {
    "put": protective_put,
    "collar": collar,
    "put_spread": put_spread,
}


def build_structure(spot: float, cfg: HedgeConfig) -> HedgeStructure:
    """Dispatch on cfg.structure. Raises ValueError on unknown structure name."""
    try:
        builder = _BUILDERS[cfg.structure]
    except KeyError as exc:
        raise ValueError(f"unknown hedge structure: {cfg.structure!r}") from exc
    return builder(spot, cfg)
