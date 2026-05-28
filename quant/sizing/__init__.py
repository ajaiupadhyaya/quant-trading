"""Composable, point-in-time position sizing — an observed, comparison-only overlay."""

from quant.sizing.backtest import SizingComparison, apply_sizing, compare_sizing
from quant.sizing.components import (
    drawdown_throttle,
    fractional_kelly,
    regime_multiplier,
    vol_target_scale,
)
from quant.sizing.models import DEFAULT_REGIME_WEIGHTS, SizingConfig, SizingDecision
from quant.sizing.policy import compute_gross

__all__ = [
    "DEFAULT_REGIME_WEIGHTS",
    "SizingComparison",
    "SizingConfig",
    "SizingDecision",
    "apply_sizing",
    "compare_sizing",
    "compute_gross",
    "drawdown_throttle",
    "fractional_kelly",
    "regime_multiplier",
    "vol_target_scale",
]
