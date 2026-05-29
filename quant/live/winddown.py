"""Governed wind-down of orphan positions: exit-only, ADV-capped, fail-closed.

An orphan = a registered slug holding a non-zero position whose governance
state is not LIVE. The owning strategy is NEVER run (it could re-open); we only
reduce its book toward flat. These helpers are pure given their inputs
(snapshot / bars / governance state) so they unit-test without Alpaca.
"""

from __future__ import annotations

import math


def capped_qty(
    order_qty: int, adv_dollar: float, price: float, participation_fraction: float
) -> int:
    """Largest share qty <= order_qty whose notional stays within
    ``participation_fraction`` of trailing dollar-ADV. Returns 0 when un-sizable
    (non-positive/non-finite ADV or price, or non-positive order qty)."""
    if order_qty <= 0 or participation_fraction <= 0.0:
        return 0
    if not (math.isfinite(adv_dollar) and math.isfinite(price)):
        return 0
    if adv_dollar <= 0.0 or price <= 0.0:
        return 0
    max_shares = int((adv_dollar * participation_fraction) / price)
    return max(0, min(order_qty, max_shares))
