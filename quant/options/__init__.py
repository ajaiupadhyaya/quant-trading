"""Options/Greeks engine + protective hedging overlay — an observed, comparison-only signal."""

from quant.options.beta import rolling_beta
from quant.options.models import (
    DEFAULT_REGIME_INTENSITY,
    HedgeConfig,
    HedgeDecision,
    HedgeStructure,
    OptionLeg,
)
from quant.options.overlay import (
    HedgeComparison,
    HedgeLedger,
    apply_hedge,
    compare_hedge,
    cvar,
    worst_day,
)
from quant.options.policy import build_hedge
from quant.options.pricing import Greeks, bs_greeks, bs_price, implied_vol
from quant.options.structures import build_structure, collar, protective_put, put_spread
from quant.options.surface import (
    OptionQuote,
    VolSurface,
    VolSurfaceConfig,
    compute_vol_surface,
    live_vol_surface,
    parse_occ_symbol,
    render_vol_surface,
)

__all__ = [
    "DEFAULT_REGIME_INTENSITY",
    "Greeks",
    "HedgeComparison",
    "HedgeConfig",
    "HedgeDecision",
    "HedgeLedger",
    "HedgeStructure",
    "OptionLeg",
    "OptionQuote",
    "VolSurface",
    "VolSurfaceConfig",
    "apply_hedge",
    "bs_greeks",
    "bs_price",
    "build_hedge",
    "build_structure",
    "collar",
    "compare_hedge",
    "compute_vol_surface",
    "cvar",
    "implied_vol",
    "live_vol_surface",
    "parse_occ_symbol",
    "protective_put",
    "put_spread",
    "render_vol_surface",
    "rolling_beta",
    "worst_day",
]
