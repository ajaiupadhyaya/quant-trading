"""Governed wind-down of orphan positions: exit-only, ADV-capped, fail-closed.

An orphan = a registered slug holding a non-zero position whose governance
state is not LIVE. The owning strategy is NEVER run (it could re-open); we only
reduce its book toward flat. These helpers are pure given their inputs
(snapshot / bars / governance state) so they unit-test without Alpaca.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from quant.backtest.impact import trailing_dollar_adv
from quant.execution.orders import OrderSide, OrderTemplate
from quant.execution.reconciler import reconcile


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


@dataclass(frozen=True)
class WindDownResult:
    """Outcome of one orphan slug's wind-down pass."""

    slug: str
    orders: list[OrderTemplate]
    reference_prices: dict[str, float]
    remaining: dict[str, int]
    skipped: list[str] = field(default_factory=list)


def _latest_close(bars: pd.DataFrame, symbol: str) -> float | None:
    col = (symbol, "close")
    if col not in bars.columns:
        return None
    series = bars[col].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def winddown_orders(
    slug: str,
    snapshot: dict[str, int],
    bars: pd.DataFrame,
    asof: date,
    participation_fraction: float,
    adv_window: int = 21,
) -> WindDownResult:
    """Exit-only orders reducing ``snapshot`` toward flat, each capped at the ADV
    participation fraction. Forces ``target={}`` into ``reconcile`` so it is
    structurally flatten-only (never opens). ``remaining`` is the post-exit book
    (current minus capped exit) INCLUDING explicit 0 for fully-exited symbols and
    the unchanged qty for un-sizable symbols, so the caller can persist a coherent
    snapshot and the orphan converges across rebalances."""
    raw = reconcile(target={}, current=snapshot, strategy_slug=slug)
    fill_ts = pd.Timestamp(asof)
    capped: list[OrderTemplate] = []
    reference_prices: dict[str, float] = {}
    skipped: list[str] = []
    remaining: dict[str, int] = {sym: int(q) for sym, q in snapshot.items()}

    for order in raw:
        sym = order.symbol
        price = _latest_close(bars, sym)
        if price is not None:
            reference_prices[sym] = price
        adv = trailing_dollar_adv(bars, sym, fill_ts, adv_window)
        cap = capped_qty(
            order.qty, adv, price if price is not None else 0.0, participation_fraction
        )
        if cap <= 0:
            skipped.append(sym)
            continue
        capped.append(OrderTemplate(symbol=sym, qty=cap, side=order.side, strategy_slug=slug))
        cur = remaining.get(sym, 0)
        remaining[sym] = cur - cap if order.side is OrderSide.SELL else cur + cap

    return WindDownResult(
        slug=slug,
        orders=capped,
        reference_prices=reference_prices,
        remaining=remaining,
        skipped=skipped,
    )


def detect_orphans(data_dir: Path) -> list[str]:
    """Sorted registered slugs whose governance state is not LIVE and which hold
    a non-zero latest snapshot. Returns [] if governance state is unavailable
    (fail-closed: no governance => no wind-down)."""
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import load_strategy_states, strategy_states_path
    from quant.live.bookkeeping import last_strategy_positions
    from quant.strategies import REGISTRY

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError:
        return []

    orphans: list[str] = []
    for slug in REGISTRY:
        state = states.get(slug)
        if state is not None and state.state is GovernanceState.LIVE:
            continue
        snap = {s: q for s, q in last_strategy_positions(data_dir, slug).items() if q != 0}
        if snap:
            orphans.append(slug)
    return sorted(orphans)
