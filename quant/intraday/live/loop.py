"""The intraday tick loop — the spine. run_tick() performs ONE lifecycle pass with
injected collaborators (no network, no sleep) so it is fully unit-testable. run_loop()
is the driver that ticks every config.tick_seconds during the session.

Lifecycle per tick (spec section 3):
  session-check -> guardrails-first (halt/loss/flat) -> pull quotes -> strategy
  -> size+caps -> reconcile vs ledger -> submit (unique COID) -> journal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from quant.governance.halt import load_halt
from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.feed import FeedError
from quant.intraday.live.guardrails import (
    clamp_qty_to_caps,
    daily_loss_breached,
    in_flat_window,
    trade_budget_exhausted,
)
from quant.intraday.live.halt import load_sleeve_halt, set_sleeve_halt
from quant.intraday.live.ids import make_sleeve_coid
from quant.intraday.live.journal import TickRecord, append_tick
from quant.intraday.live.sleeve import Fill, SleeveLedger
from quant.intraday.strategy import IntradayStrategy, Order, Side


class _Broker(Protocol):
    def account(self) -> Any: ...
    def submit_simple_order(self, *, symbol: str, side: str, qty: int,
                            client_order_id: str, order_type: str = "market",
                            limit_price: float | None = None, dry_run: bool = False) -> str: ...


class _Feed(Protocol):
    def latest_quotes(self, now: datetime | None = None) -> list[QuoteBar]: ...


@dataclass
class TickDeps:
    data_dir: Path
    config: SleeveConfig
    broker: _Broker
    feed: _Feed
    strategy: IntradayStrategy
    ledger: SleeveLedger
    now: datetime
    session_open: bool
    session_close: datetime
    dry_run: bool = False


class _Ctx:
    """StrategyContext backed by the ledger + this tick's marks."""

    def __init__(self, ledger: SleeveLedger, marks: dict[str, float], ts: datetime) -> None:
        self._ledger = ledger
        self._marks = marks
        self._ts = ts

    def position(self, symbol: str) -> int:
        return self._ledger.position(symbol)

    def cash(self) -> float:
        return 0.0

    def nbbo(self, symbol: str) -> QuoteBar | None:
        return None

    def now(self) -> datetime:
        return self._ts


def _flatten_all(deps: TickDeps, marks: dict[str, float]) -> int:
    """Submit market orders closing every open sleeve position. Returns order count."""
    n = 0
    for sym, pos in list(deps.ledger.positions().items()):
        side = "sell" if pos > 0 else "buy"
        coid = make_sleeve_coid(sym, deps.now, n)
        deps.broker.submit_simple_order(
            symbol=sym, side=side, qty=abs(pos),
            client_order_id=coid, dry_run=deps.dry_run,
        )
        # Record the closing fill in the ledger (modelled at mid-price).
        signed = -pos  # closing is opposite sign of position
        deps.ledger.record(Fill(symbol=sym, qty=signed, price=marks.get(sym, 0.0)))
        n += 1
    return n


def run_tick(deps: TickDeps) -> None:
    """Execute one full lifecycle pass: guardrails-first then strategy+execution."""
    cfg = deps.config

    # 1. Session check — idle if market is closed.
    if not deps.session_open:
        logger.debug("intraday loop idle (session closed)")
        return

    # 2a. Halt check (global governance + sleeve-local) — checked BEFORE any action.
    if load_halt(deps.data_dir).active or load_sleeve_halt(deps.data_dir).active:
        logger.warning("intraday loop halted; skipping tick")
        return

    # 3. Pull quotes — never trade on stale/missing data.
    try:
        bars = deps.feed.latest_quotes(deps.now)
    except FeedError as exc:
        logger.warning("feed error, skipping tick actions: {}", exc)
        return
    marks: dict[str, float] = {b.symbol: b.mid for b in bars}

    allocation = cfg.sleeve_allocation(float(deps.broker.account().equity))
    day_pnl = deps.ledger.day_pnl(marks)

    # 2b. Daily loss kill-switch — flatten + halt BEFORE any new entry.
    if daily_loss_breached(day_pnl=day_pnl, sleeve_allocation=allocation, config=cfg):
        n = _flatten_all(deps, marks)
        set_sleeve_halt(deps.data_dir, reason=f"daily loss breach: day_pnl={day_pnl:.2f}")
        _journal(deps, marks, n_orders=n, halted=True, note="loss-halt")
        return

    # 2c. Flat-by-close window — flatten all and skip new entries.
    if in_flat_window(deps.now, deps.session_close, cfg):
        n = _flatten_all(deps, marks)
        _journal(deps, marks, n_orders=n, halted=False, note="flat-by-close")
        return

    # 4. Strategy targets — feed each quote bar as an event.
    ctx = _Ctx(deps.ledger, marks, deps.now)
    desired: list[Order] = []
    for b in bars:
        desired.extend(deps.strategy.on_event(b, ctx))

    # 5-7. Size, cap, reconcile, submit.
    n_orders = 0
    for order in desired:
        # Trade-budget guard applies only to new opens, not to exits.
        is_new_open = deps.ledger.position(order.symbol) == 0
        if is_new_open and trade_budget_exhausted(round_trips=deps.ledger.round_trips, config=cfg):
            continue
        price = marks.get(order.symbol)
        if price is None:
            continue
        qty = clamp_qty_to_caps(
            desired_qty=order.qty,
            price=price,
            gross_notional=deps.ledger.gross_notional(marks),
            sleeve_allocation=allocation,
            config=cfg,
        )
        if qty <= 0:
            continue
        side_str = "buy" if order.side is Side.BUY else "sell"
        coid = make_sleeve_coid(order.symbol, deps.now, n_orders)
        deps.broker.submit_simple_order(
            symbol=order.symbol, side=side_str, qty=qty,
            client_order_id=coid, dry_run=deps.dry_run,
        )
        # Model the fill at quote mid for the internal ledger.
        signed_qty = qty if order.side is Side.BUY else -qty
        deps.ledger.record(Fill(symbol=order.symbol, qty=signed_qty, price=price))
        n_orders += 1

    _journal(deps, marks, n_orders=n_orders, halted=False, note="ok")


def _journal(
    deps: TickDeps,
    marks: dict[str, float],
    *,
    n_orders: int,
    halted: bool,
    note: str,
) -> None:
    append_tick(
        deps.data_dir,
        TickRecord(
            ts=deps.now,
            sleeve_value=deps.ledger.gross_notional(marks),
            day_pnl=deps.ledger.day_pnl(marks),
            round_trips=deps.ledger.round_trips,
            n_orders=n_orders,
            halted=halted,
            note=note,
        ),
    )


def run_loop(
    deps_factory: Any,
    *,
    max_ticks: int | None = None,
    sleep_s: float | None = None,
) -> None:
    """Driver: build fresh TickDeps each tick via deps_factory() and run_tick().

    deps_factory is a zero-arg callable returning a TickDeps for 'now'. max_ticks
    bounds runs (None = forever); sleep_s overrides config cadence (tests pass 0).
    """
    count = 0
    while max_ticks is None or count < max_ticks:
        deps: TickDeps = deps_factory()
        try:
            run_tick(deps)
        except Exception:
            logger.exception("intraday tick failed; continuing")
        count += 1
        time.sleep(sleep_s if sleep_s is not None else deps.config.tick_seconds)
