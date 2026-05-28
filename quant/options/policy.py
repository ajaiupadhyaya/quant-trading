"""Compose beta + regime intensity + structure into a HedgeDecision."""

from __future__ import annotations

import numpy as np

from quant.options.beta import rolling_beta
from quant.options.models import HedgeConfig, HedgeDecision, HedgeStructure
from quant.options.structures import build_structure

_TRADING_DAYS = 252.0
_FALLBACK_VOL = 0.15


def _intensity(regime_label: str | None, cfg: HedgeConfig) -> float:
    if not cfg.use_regime or regime_label is None:
        return 1.0
    return float(cfg.regime_intensity.get(regime_label, 1.0))


def _trailing_vol(window: np.ndarray) -> float:
    arr = np.asarray(window, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return _FALLBACK_VOL
    daily = float(np.std(arr, ddof=1))
    ann = daily * float(np.sqrt(_TRADING_DAYS))
    return ann if ann > 1e-6 else _FALLBACK_VOL


def build_hedge(
    spot: float,
    book_returns_hist: np.ndarray,
    spy_returns_hist: np.ndarray,
    regime_label: str | None,
    cfg: HedgeConfig,
    book_value: float,
    expiry_index: int,
) -> HedgeDecision:
    """Build one roll's hedge. All inputs are strictly trailing (PIT)."""
    beta = rolling_beta(
        np.asarray(book_returns_hist, dtype=float)[-cfg.beta_lookback_days :],
        np.asarray(spy_returns_hist, dtype=float)[-cfg.beta_lookback_days :],
    )
    intensity = _intensity(regime_label, cfg)
    base = build_structure(spot, cfg)
    structure = HedgeStructure(legs=base.legs, spot_at_open=spot, expiry_index=expiry_index)

    contracts = cfg.coverage * intensity * beta * book_value / spot if spot > 0 else 0.0

    tenor_years = cfg.tenor_days / 365.0
    vol = _trailing_vol(np.asarray(spy_returns_hist, dtype=float)[-cfg.vol_lookback_days :])
    unit_premium = structure.value(spot, tenor_years, vol, cfg.risk_free, cfg.div_yield)
    premium = contracts * unit_premium

    return HedgeDecision(
        structure=structure,
        contracts=contracts,
        premium=premium,
        net_beta=beta,
        regime_label=regime_label,
        intensity=intensity,
    )
