"""Pure guardrail decision functions. No I/O, no broker calls — the loop composes
these. Every function is independently testable."""

from __future__ import annotations

from datetime import datetime, timedelta

from quant.intraday.live.config import SleeveConfig


def clamp_qty_to_caps(
    *,
    desired_qty: int,
    price: float,
    gross_notional: float,
    sleeve_allocation: float,
    config: SleeveConfig,
) -> int:
    """Clamp |desired_qty| to BOTH the per-trade cap and the remaining sleeve room.
    Returns a non-negative share count (sign handled by the caller)."""
    if price <= 0 or desired_qty <= 0:
        return 0
    per_trade_shares = int(config.per_trade_cap // price)
    room_dollars = max(0.0, sleeve_allocation - gross_notional)
    room_shares = int(room_dollars // price)
    return max(0, min(desired_qty, per_trade_shares, room_shares))


def trade_budget_exhausted(*, round_trips: int, config: SleeveConfig) -> bool:
    return round_trips >= config.max_round_trips


def daily_loss_breached(*, day_pnl: float, sleeve_allocation: float, config: SleeveConfig) -> bool:
    threshold = -abs(config.daily_loss_halt_pct) * sleeve_allocation
    return day_pnl <= threshold


def in_flat_window(now: datetime, session_close: datetime, config: SleeveConfig) -> bool:
    return now >= session_close - timedelta(minutes=config.flat_by_close_minutes)
