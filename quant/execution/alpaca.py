"""Alpaca client wrapper.

Thin layer over alpaca-py's TradingClient that:
- normalizes string-typed API responses into typed dataclasses,
- attaches client_order_id with per-strategy attribution,
- supports dry-run order submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest

from quant.execution.orders import (
    OrderSide,
    OrderTemplate,
    OrderType,
    TimeInForce,
    make_client_order_id,
)
from quant.util.config import Settings
from quant.util.logging import logger

_TIF_MAP = {TimeInForce.DAY: AlpacaTIF.DAY, TimeInForce.GTC: AlpacaTIF.GTC}


@dataclass(frozen=True)
class AccountInfo:
    equity: float
    last_equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    pattern_day_trader: bool


@dataclass(frozen=True)
class PositionRow:
    symbol: str
    qty: int
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    current_price: float
    side: str  # "long" or "short"


@dataclass(frozen=True)
class OrderRow:
    """A single Alpaca order with fill outcome, in plain Python types."""

    client_order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    submitted_qty: int
    filled_qty: int
    filled_avg_price: float | None
    submitted_at: datetime
    filled_at: datetime | None
    status: str  # alpaca-py OrderStatus.value, e.g. "filled" | "canceled" | "rejected"


def _f(x: object) -> float:
    return float(x) if x is not None else 0.0  # type: ignore[arg-type]


def _i(x: object) -> int:
    return int(float(x)) if x is not None else 0  # type: ignore[arg-type]


class AlpacaClient:
    """Wraps `alpaca-py` for the subset of operations we need."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()  # type: ignore[call-arg]
        self._trading = TradingClient(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            paper=self.settings.alpaca_paper,
        )

    def account(self) -> AccountInfo:
        raw = self._trading.get_account()
        return AccountInfo(
            equity=_f(raw.equity),  # type: ignore[union-attr]
            last_equity=_f(raw.last_equity),  # type: ignore[union-attr]
            buying_power=_f(raw.buying_power),  # type: ignore[union-attr]
            cash=_f(raw.cash),  # type: ignore[union-attr]
            portfolio_value=_f(raw.portfolio_value),  # type: ignore[union-attr]
            pattern_day_trader=bool(raw.pattern_day_trader),  # type: ignore[union-attr]
        )

    def positions(self) -> list[PositionRow]:
        raw_positions = self._trading.get_all_positions()
        rows: list[PositionRow] = []
        for p in raw_positions:
            side = str(p.side).lower()  # type: ignore[union-attr]
            qty = _i(p.qty)  # type: ignore[union-attr]
            if side == "short":
                qty = -abs(qty)
            rows.append(
                PositionRow(
                    symbol=str(p.symbol),  # type: ignore[union-attr]
                    qty=qty,
                    avg_entry_price=_f(p.avg_entry_price),  # type: ignore[union-attr]
                    market_value=_f(p.market_value),  # type: ignore[union-attr]
                    unrealized_pl=_f(p.unrealized_pl),  # type: ignore[union-attr]
                    current_price=_f(p.current_price),  # type: ignore[union-attr]
                    side=side,
                )
            )
        return rows

    def list_orders(
        self,
        *,
        since: date,
        until: date,
        limit: int = 500,
    ) -> list[OrderRow]:
        """Fetch orders submitted on [since, until] inclusive, newest first."""
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=datetime.combine(since, datetime.min.time(), tzinfo=UTC),
            until=datetime.combine(until, datetime.max.time(), tzinfo=UTC),
            limit=limit,
        )
        orders = self._trading.get_orders(filter=req)
        rows: list[OrderRow] = []
        for o in orders:
            filled_avg = o.filled_avg_price  # type: ignore[union-attr]
            rows.append(
                OrderRow(
                    client_order_id=str(o.client_order_id),  # type: ignore[union-attr]
                    symbol=str(o.symbol),  # type: ignore[union-attr]
                    side=str(o.side.value),  # type: ignore[union-attr]
                    submitted_qty=_i(o.qty),  # type: ignore[union-attr]
                    filled_qty=_i(o.filled_qty),  # type: ignore[union-attr]
                    filled_avg_price=_f(filled_avg) if filled_avg is not None else None,
                    submitted_at=o.submitted_at,  # type: ignore[union-attr]
                    filled_at=o.filled_at,  # type: ignore[union-attr]
                    status=str(o.status.value),  # type: ignore[union-attr]
                )
            )
        return rows

    def list_orders_for_date(self, asof: date) -> list[OrderRow]:
        """Orders submitted on the single calendar day ``asof`` (00:00..23:59 UTC).

        The idempotency guard (``quant.live.rebalance.already_traded_today``)
        duck-types this method off the client; it must exist on the real client
        so reconcile-then-refuse actually queries Alpaca in production, not only
        when a test injects a fake.
        """
        return self.list_orders(since=asof, until=asof)

    def submit_order(
        self, order: OrderTemplate, *, asof: date | None = None, dry_run: bool = False
    ) -> str:
        """Submit the order described by ``order``. Returns the client_order_id.

        Honors the template's ``order_type``/``limit_price``/``time_in_force``.
        Defaults (MARKET / DAY / no limit) reproduce the historical market+DAY
        request byte-for-byte; no live path emits non-default fields today.

        ``asof`` is the rebalance session date stamped into the deterministic
        client_order_id; it must match the date the idempotency guard
        (``already_traded_today``) queries, so a same-day retry collides on the
        COID and the broker rejects the duplicate. Defaults to today.
        """
        coid = make_client_order_id(order.strategy_slug, order.symbol, asof or date.today())
        side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL
        tif = _TIF_MAP[order.time_in_force]
        req: MarketOrderRequest | LimitOrderRequest
        if order.order_type is OrderType.LIMIT:
            req = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=coid,
                limit_price=order.limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=coid,
            )
        if dry_run:
            logger.info(
                "[DRY-RUN] would submit {} {} {} {}{} (coid={})",
                order.side,
                order.qty,
                order.symbol,
                order.order_type,
                f"@{order.limit_price}" if order.limit_price is not None else "",
                coid,
            )
            return coid
        self._trading.submit_order(req)
        logger.info("Submitted {} {} {} (coid={})", order.side, order.qty, order.symbol, coid)
        return coid

    def submit_simple_order(
        self,
        *,
        symbol: str,
        side: str,  # "buy" | "sell"
        qty: int,
        client_order_id: str,
        order_type: str = "market",  # "market" | "limit"
        limit_price: float | None = None,
        dry_run: bool = False,
    ) -> str:
        """Submit a single intraday order with a CALLER-SUPPLIED client_order_id.

        Unlike submit_order (deterministic per-day COID for idempotent daily
        rebalances), the intraday loop places many same-symbol orders per day, so
        the COID must be unique per order — the caller owns uniqueness.
        Time-in-force is DAY. Returns the client_order_id.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        alp_side = AlpacaSide.BUY if side == "buy" else AlpacaSide.SELL
        req: MarketOrderRequest | LimitOrderRequest
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit order requires limit_price")
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alp_side,
                time_in_force=AlpacaTIF.DAY,
                client_order_id=client_order_id,
                limit_price=limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alp_side,
                time_in_force=AlpacaTIF.DAY,
                client_order_id=client_order_id,
            )
        if dry_run:
            logger.info(
                "[DRY-RUN] sleeve would submit {} {} {} (coid={})",
                side,
                qty,
                symbol,
                client_order_id,
            )
            return client_order_id
        self._trading.submit_order(req)
        logger.info("Sleeve submitted {} {} {} (coid={})", side, qty, symbol, client_order_id)
        return client_order_id
