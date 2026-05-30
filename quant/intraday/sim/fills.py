"""Fill model: marketable (spread-crossing) + limit (fills only when the 1s NBBO
trades through). Pure functions; all costs charged here. The engine supplies
adv_dollar per symbol (so this stays decoupled from the daily bars frame)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant.backtest.impact import market_impact_bps
from quant.intraday.data.events import QuoteBar, Trade
from quant.intraday.strategy import Order, Side


@dataclass(frozen=True)
class Fill:
    ts: datetime
    symbol: str
    side: Side
    qty: int
    price: float
    commission: float
    impact_cost: float
    spread_cost: float

    @property
    def signed_qty(self) -> int:
        return self.qty if self.side is Side.BUY else -self.qty


def marketable_fill(
    order: Order,
    nbbo: QuoteBar | None,
    ts: datetime,
    *,
    adv_dollar: float,
    impact_coef_bps: float,
    commission_per_share: float,
) -> Fill | None:
    """Cross the spread at the far touch; add square-root impact + commission."""
    if nbbo is None:
        return None
    ref = nbbo.ask if order.side is Side.BUY else nbbo.bid
    notional = ref * order.qty
    imp_bps = market_impact_bps(notional, adv_dollar, impact_coef_bps)
    slip = ref * (imp_bps / 1e4)
    price = ref + slip if order.side is Side.BUY else ref - slip
    spread_per_share = abs(ref - nbbo.mid)
    return Fill(
        ts=ts,
        symbol=order.symbol,
        side=order.side,
        qty=order.qty,
        price=price,
        commission=commission_per_share * order.qty,
        impact_cost=slip * order.qty,
        spread_cost=spread_per_share * order.qty,
    )


def limit_crosses(order: Order, event: Trade | QuoteBar) -> bool:
    """True if this event shows the market trading THROUGH the limit (conservative;
    no passive-queue credit). Buy fills when price/ask <= limit; sell when
    price/bid >= limit."""
    if order.limit_price is None:
        return False
    if isinstance(event, Trade):
        ref = event.price
        return ref <= order.limit_price if order.side is Side.BUY else ref >= order.limit_price
    if order.side is Side.BUY:
        return event.ask <= order.limit_price
    return event.bid >= order.limit_price


def limit_fill(
    order: Order,
    event: Trade | QuoteBar,
    ts: datetime,
    *,
    commission_per_share: float,
) -> Fill | None:
    """Fill a resting limit at the limit price iff the event trades through it."""
    if not limit_crosses(order, event):
        return None
    assert order.limit_price is not None
    return Fill(
        ts=ts,
        symbol=order.symbol,
        side=order.side,
        qty=order.qty,
        price=order.limit_price,
        commission=commission_per_share * order.qty,
        impact_cost=0.0,
        spread_cost=0.0,
    )
