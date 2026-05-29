"""Net per-symbol orders so opposing orders never reach the broker.

The account is a single shared Alpaca account: when one strategy buys a symbol
another sells, the orders oppose and the broker rejects the second (wash-trade).
Netting collapses all intended orders into one net order per symbol = the
account's desired delta. Per-strategy snapshots remain the unit of intent; this
only changes order SUBMISSION.
"""

from __future__ import annotations

from collections import defaultdict

from quant.execution.orders import OrderSide, OrderTemplate


def net_orders(orders: list[OrderTemplate]) -> list[OrderTemplate]:
    """Collapse orders into one net OrderTemplate per symbol.

    Signed qty: BUY=+qty, SELL=-qty. net>0 -> BUY |net|; net<0 -> SELL |net|;
    net==0 -> omitted (fully offsetting). The net order's strategy_slug is the
    slug contributing the largest absolute qty to that symbol (ties broken
    alphabetically); attribution is for the trade log / client_order_id only.
    Output is sorted by symbol for deterministic client_order_ids.
    """
    signed: dict[str, int] = defaultdict(int)
    contrib: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for o in orders:
        signed[o.symbol] += o.qty if o.side is OrderSide.BUY else -o.qty
        contrib[o.symbol][o.strategy_slug] += abs(o.qty)

    out: list[OrderTemplate] = []
    for symbol in sorted(signed):
        net = signed[symbol]
        if net == 0:
            continue
        side = OrderSide.BUY if net > 0 else OrderSide.SELL
        owner = min(contrib[symbol].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        out.append(OrderTemplate(symbol=symbol, qty=abs(net), side=side, strategy_slug=owner))
    return out
