"""Deterministic one-way forecast-vol-target overlay for the live rebalance.

Anticipatory risk control: when the OOS-validated vol forecast says the book's
next-day volatility will exceed its recent trailing level, scale total gross down
toward target. Mirrors the de-risk overlay's contract exactly — a portfolio-level
gross MULTIPLIER the rebalance applies on top of the strategies (it does NOT touch
their internal sizing, so their validated evidence stands; see
docs/specs/2026-06-10-forecast-driven-vol-targeting.md, gate-passed A/B).

Contract:
- ``actuate=False`` (default) ⇒ SHADOW: the multiplier is computed and reported
  but NOT applied; the rebalance is byte-for-byte today's behavior.
- De-risk-ONLY: ``cap=1.0`` means the factor can only SHRINK gross (never lever
  up), clamped to ``floor``. Fully reversible (a calmer forecast restores size).
- Fail-SAFE: insufficient history / a degenerate forecast contributes NO change
  (multiplier 1.0). The only failure direction is *less* de-risk, never more.
- Pure + deterministic given the returns input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quant.forecast.vol import forecast_vol_ann_next


@dataclass(frozen=True)
class VolTargetConfig:
    """Knobs for the forecast-vol-target overlay. Inert by default (``actuate=False``)."""

    actuate: bool = False
    cap: float = 1.0  # de-risk-only: never lever above full gross
    floor: float = 0.5  # never cut below this fraction in one rebalance step
    vol_lookback_days: int = 63  # trailing window defining the book's "current vol" target
    forecast_model: str = "gjr"  # gjr -> har -> ewma cascade
    min_history_days: int = 252  # below this the forecast degrades ⇒ no-op


@dataclass(frozen=True)
class VolTargetResult:
    multiplier: float  # computed one-way factor in [floor, cap]
    applied: float  # what the rebalance uses: multiplier if actuated else 1.0
    actuated: bool
    forecast_vol_ann: float | None
    trailing_vol_ann: float | None
    reasons: list[str] = field(default_factory=list)
    degraded: bool = False  # insufficient history / degenerate forecast ⇒ no change


def voltarget_multiplier(equity_returns: np.ndarray, cfg: VolTargetConfig) -> VolTargetResult:
    """One-way forecast-vol-target factor in ``[floor, cap]`` from the book's returns.

    ``target = trailing realized vol`` (the book's current level); ``factor =
    min(cap, target / forecast_vol)`` clamped to ``floor``. So the overlay only
    cuts when the forecast vol exceeds the recent trailing vol (anticipatory
    de-risk). ``applied`` is the factor when ``cfg.actuate`` else 1.0 (shadow).
    Any degenerate input ⇒ ``multiplier=1.0`` (fail-safe, no de-risk).
    """
    r = np.asarray(equity_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < cfg.min_history_days:
        return VolTargetResult(1.0, 1.0, cfg.actuate, None, None, ["insufficient-history"], True)

    tail = r[-cfg.vol_lookback_days :]
    trailing = float(np.std(tail, ddof=1) * math.sqrt(252)) if tail.size >= 2 else 0.0
    forecast = forecast_vol_ann_next(r, model=cfg.forecast_model, min_obs=cfg.vol_lookback_days)

    if forecast is None or not math.isfinite(forecast) or forecast <= 0.0 or trailing <= 0.0:
        return VolTargetResult(
            1.0, 1.0, cfg.actuate, forecast, trailing or None, ["degraded"], True
        )

    factor = min(cfg.cap, trailing / forecast)
    multiplier = max(cfg.floor, round(factor, 4))
    reasons: list[str] = []
    if multiplier < 1.0:
        reasons.append(f"forecast_vol={forecast:.3f}>trailing={trailing:.3f}(x{multiplier})")
    applied = multiplier if cfg.actuate else 1.0
    return VolTargetResult(multiplier, applied, cfg.actuate, forecast, trailing, reasons, False)


def to_report_dict(result: VolTargetResult) -> dict[str, Any]:
    """Serializable shadow payload for the rebalance report / artifact."""
    return {
        "multiplier": result.multiplier,
        "applied": result.applied,
        "actuated": result.actuated,
        "forecast_vol_ann": result.forecast_vol_ann,
        "trailing_vol_ann": result.trailing_vol_ann,
        "reasons": list(result.reasons),
        "degraded": result.degraded,
    }
