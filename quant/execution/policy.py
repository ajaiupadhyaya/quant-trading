"""Impact-aware live execution policy.

Adjusts the *netted* orders at submission time so the live executor respects the
same participation/impact notion the backtest already charges for
(``quant.backtest.impact``), instead of dumping market orders of arbitrary size.

The system is a daily batch, so there is no intraday loop to work an order.
Participation control + the daily target-vs-current reconcile loop together give
multi-session slicing "for free": an oversized order is capped today and its
residual is re-proposed (and submitted) on subsequent sessions.

Contract:
- ``enabled=False`` ⇒ identity transform, byte-for-byte today's behavior.
- Fail-open: a symbol whose dollar-ADV or reference price is unknown / degenerate
  is passed through unchanged. The policy NEVER blocks a trade because it cannot
  estimate impact.

Pure functions only — no I/O, no config-file or Settings import — so the
rebalance path constructs the config explicitly (mirroring ``PortfolioRiskLimits``
at the Guard-5 gate) and tests stay deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from quant.execution.orders import OrderSide, OrderTemplate, OrderType


@dataclass(frozen=True)
class ExecutionPolicyConfig:
    """Parameters for the impact-aware execution policy. All knobs live here.

    Defaults are inert by design: ``enabled=False`` means the policy is an
    identity transform. The numeric defaults are only consulted once a caller
    consciously enables it.
    """

    enabled: bool = False
    max_participation: float = 0.10
    adv_window: int = 21
    marketable_limit_bps: float | None = None
    marketable_threshold: float = 0.05


def participation(notional: float, dollar_adv: float) -> float | None:
    """Order notional as a fraction of trailing dollar-ADV.

    Returns ``None`` when impact cannot be estimated: non-finite notional, or
    non-finite / non-positive ADV. (Mirrors ``backtest.impact`` tolerance.)
    """
    if not math.isfinite(notional):
        return None
    if not math.isfinite(dollar_adv) or dollar_adv <= 0.0:
        return None
    return notional / dollar_adv


def cap_qty_to_participation(
    qty: int, ref_price: float, dollar_adv: float, cfg: ExecutionPolicyConfig
) -> tuple[int, int]:
    """Cap ``qty`` so its notional ≤ ``max_participation`` of dollar-ADV.

    Returns ``(capped_qty, deferred_qty)`` with ``capped_qty + deferred_qty ==
    qty`` (both ≥ 0). Fail-open: if ADV or ``ref_price`` is unknown / degenerate
    the order passes through uncapped (``(qty, 0)``) — impact cannot be estimated,
    so we do not penalize the trade.
    """
    if not math.isfinite(dollar_adv) or dollar_adv <= 0.0:
        return qty, 0
    if not math.isfinite(ref_price) or ref_price <= 0.0:
        return qty, 0
    max_notional = cfg.max_participation * dollar_adv
    max_qty = math.floor(max_notional / ref_price)
    capped = min(qty, max(max_qty, 0))
    return capped, qty - capped


def marketable_limit_price(
    side: OrderSide, ref_price: float, cfg: ExecutionPolicyConfig
) -> float | None:
    """Marketable limit price ``ref·(1 ± bps)`` (buy ⇒ above, sell ⇒ below).

    Caps adverse slippage while staying marketable. Returns ``None`` when no
    ``marketable_limit_bps`` is configured or ``ref_price`` is degenerate
    (⇒ keep the order MARKET). Rounded to cents for broker acceptance.
    """
    if cfg.marketable_limit_bps is None:
        return None
    if not math.isfinite(ref_price) or ref_price <= 0.0:
        return None
    edge = cfg.marketable_limit_bps / 10_000.0
    raw = ref_price * (1.0 + edge) if side is OrderSide.BUY else ref_price * (1.0 - edge)
    return round(raw, 2)


def apply_execution_policy(
    orders: list[OrderTemplate],
    *,
    dollar_adv: dict[str, float],
    reference_prices: dict[str, float],
    cfg: ExecutionPolicyConfig,
) -> tuple[list[OrderTemplate], list[dict[str, Any]]]:
    """Adjust netted ``orders`` for participation/impact. The only entry point.

    Returns ``(adjusted_orders, plan_rows)``. When ``cfg.enabled`` is ``False``
    the input list is returned unchanged (same objects) with no plan rows.

    Fully-deferred orders (cap → 0) are dropped from this session; per-strategy
    bookkeeping already recorded the *target*, so the residual is re-proposed and
    submitted on the next reconcile. ``plan_rows`` is the execution-plan artifact
    payload (one row per order the policy considered).
    """
    if not cfg.enabled:
        return list(orders), []

    adjusted: list[OrderTemplate] = []
    rows: list[dict[str, Any]] = []
    for order in orders:
        ref = reference_prices.get(order.symbol, float("nan"))
        adv = dollar_adv.get(order.symbol, float("nan"))
        capped, deferred = cap_qty_to_participation(order.qty, ref, adv, cfg)

        submit_notional = capped * ref if math.isfinite(ref) else float("nan")
        part = participation(submit_notional, adv)

        order_type = OrderType.MARKET
        limit_price: float | None = None
        if (
            capped > 0
            and cfg.marketable_limit_bps is not None
            and part is not None
            and part >= cfg.marketable_threshold
        ):
            lp = marketable_limit_price(order.side, ref, cfg)
            if lp is not None:
                order_type = OrderType.LIMIT
                limit_price = lp

        rows.append(
            {
                "symbol": order.symbol,
                "strategy": order.strategy_slug,
                "side": str(order.side),
                "original_qty": order.qty,
                "capped_qty": capped,
                "deferred_qty": deferred,
                "participation": part,
                "order_type": str(order_type),
                "limit_price": limit_price,
                "reason": _reason(order.qty, capped, order_type),
            }
        )

        if capped == 0:
            continue  # fully deferred this session
        if capped == order.qty and order_type is OrderType.MARKET:
            adjusted.append(order)  # untouched — keep the original object
        else:
            adjusted.append(
                replace(order, qty=capped, order_type=order_type, limit_price=limit_price)
            )

    return adjusted, rows


def _reason(original: int, capped: int, order_type: OrderType) -> str:
    if capped == 0:
        return "deferred_all"
    parts = []
    if capped < original:
        parts.append("participation_capped")
    if order_type is OrderType.LIMIT:
        parts.append("marketable_limit")
    return "+".join(parts) if parts else "passthrough"
