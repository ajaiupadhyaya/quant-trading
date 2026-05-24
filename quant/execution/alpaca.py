"""Alpaca client wrapper.

Thin layer over alpaca-py's TradingClient that:
- normalizes string-typed API responses into typed dataclasses,
- attaches client_order_id with per-strategy attribution,
- supports dry-run order submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from quant.execution.orders import OrderSide, OrderTemplate, make_client_order_id
from quant.util.config import Settings
from quant.util.logging import logger


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

    def submit_order(self, order: OrderTemplate, *, dry_run: bool = False) -> str:
        """Submit a market order. Returns the client_order_id."""
        coid = make_client_order_id(order.strategy_slug, order.symbol, date.today())
        side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
        )
        if dry_run:
            logger.info(
                "[DRY-RUN] would submit {} {} {} (coid={})",
                order.side,
                order.qty,
                order.symbol,
                coid,
            )
            return coid
        self._trading.submit_order(req)
        logger.info("Submitted {} {} {} (coid={})", order.side, order.qty, order.symbol, coid)
        return coid
