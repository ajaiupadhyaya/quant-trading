"""Compute the list of OrderTemplates needed to move from current to target positions."""

from __future__ import annotations

from quant.execution.orders import OrderSide, OrderTemplate


def reconcile(
    target: dict[str, int],
    current: dict[str, int],
    strategy_slug: str,
) -> list[OrderTemplate]:
    """Return the orders that transform `current` into `target`.

    Long-to-short or short-to-long flips are split into two orders (flatten, then
    reopen on the other side) so each fill is monotonically directional. Some
    brokers reject single orders that cross zero; this keeps us safe.
    """
    orders: list[OrderTemplate] = []
    symbols = sorted(set(target) | set(current))

    for sym in symbols:
        tgt = target.get(sym, 0)
        cur = current.get(sym, 0)
        if tgt == cur:
            continue

        # Crossing zero?
        if (cur > 0 and tgt < 0) or (cur < 0 and tgt > 0):
            # Step 1: flatten current
            flatten_side = OrderSide.SELL if cur > 0 else OrderSide.BUY
            orders.append(
                OrderTemplate(
                    symbol=sym, qty=abs(cur), side=flatten_side, strategy_slug=strategy_slug
                )
            )
            # Step 2: open target on the other side
            open_side = OrderSide.BUY if tgt > 0 else OrderSide.SELL
            orders.append(
                OrderTemplate(symbol=sym, qty=abs(tgt), side=open_side, strategy_slug=strategy_slug)
            )
            continue

        delta = tgt - cur
        if delta > 0:
            # Need to increase long exposure (or reduce short)
            orders.append(
                OrderTemplate(
                    symbol=sym, qty=abs(delta), side=OrderSide.BUY, strategy_slug=strategy_slug
                )
            )
        else:
            orders.append(
                OrderTemplate(
                    symbol=sym, qty=abs(delta), side=OrderSide.SELL, strategy_slug=strategy_slug
                )
            )

    return orders
