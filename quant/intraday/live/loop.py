"""The intraday tick loop — the spine. run_tick() performs ONE lifecycle pass with
injected collaborators (no network, no sleep) so it is fully unit-testable. run_loop()
is the driver that ticks every config.tick_seconds during the session.

Lifecycle per tick (spec section 3):
  session-check -> halt-check -> pull quotes -> loss-halt -> flat-by-close
  -> strategy -> size/cap/submit -> journal.
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
from quant.intraday.execution.calibrate import calibrate
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.manager import ExecutionManager
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
    def positions(self) -> list[Any]: ...
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
    tick_index: int = 0
    exec_manager: ExecutionManager | None = None
    exec_config: ExecConfig | None = None


def _sleeve_adv_dollar(symbol: str, price: float) -> float:
    # Mega-liquid ETF proxy ADV ($). Calibration only needs an order-of-magnitude anchor.
    _ = symbol, price  # unused: single anchor value for all symbols
    return 5_000_000_000.0  # TODO(data-layer): real trailing dollar-ADV per symbol


def _recent_returns(deps: TickDeps, symbol: str) -> list[float]:
    # Spine keeps no return history yet -> empty list -> sigma=0 (A-C degenerates to
    # TWAP-like, the safe default). A future task can wire the data layer here.
    _ = deps, symbol  # unused until the data layer wires in return history
    return []  # TODO(data-layer): real recent intraday returns (empty -> sigma=0 -> TWAP-safe)


class _Ctx:
    """StrategyContext backed by the ledger + this tick's marks and quotes."""

    def __init__(
        self,
        ledger: SleeveLedger,
        marks: dict[str, float],
        quotes: dict[str, QuoteBar],
        ts: datetime,
    ) -> None:
        self._ledger = ledger
        self._marks = marks
        self._quotes = quotes
        self._ts = ts

    def position(self, symbol: str) -> int:
        return self._ledger.position(symbol)

    def cash(self) -> float:
        return 0.0

    def nbbo(self, symbol: str) -> QuoteBar | None:
        return self._quotes.get(symbol)

    def now(self) -> datetime:
        return self._ts


def _flatten_all(deps: TickDeps, marks: dict[str, float]) -> int:
    """Submit market orders closing every open sleeve position. Returns order count."""
    n = 0
    for sym, pos in list(deps.ledger.positions().items()):
        if deps.exec_manager is not None:
            deps.exec_manager.cancel(sym)
        side = "sell" if pos > 0 else "buy"
        coid = make_sleeve_coid(sym, deps.now, n)
        deps.broker.submit_simple_order(
            symbol=sym, side=side, qty=abs(pos),
            client_order_id=coid, dry_run=deps.dry_run,
        )
        # Record the closing fill in the ledger (modelled at mid-price; fall back to
        # avg cost so P&L is ~0 for unmarked symbols rather than a total-loss artefact).
        signed = -pos  # closing is opposite sign of position
        deps.ledger.record(Fill(symbol=sym, qty=signed, price=marks.get(sym, deps.ledger.avg_cost(sym))))
        n += 1
    return n


def _work_active_programs(
    deps: TickDeps, marks: dict[str, float], allocation: float
) -> int:
    """Submit due child slices for all in-flight execution programs. Returns the
    number of slices submitted (for the tick's n_orders journal count)."""
    mgr = deps.exec_manager
    if mgr is None:
        return 0
    cfg = deps.config
    submitted = 0
    for i, child in enumerate(mgr.due_slices(deps.tick_index)):
        price = marks.get(child.symbol)
        if price is None:
            continue
        qty = clamp_qty_to_caps(
            desired_qty=child.qty, price=price,
            gross_notional=deps.ledger.gross_notional(marks),
            sleeve_allocation=allocation, config=cfg,
        )
        if qty <= 0:
            continue
        side_str = "buy" if child.side is Side.BUY else "sell"
        # offset by 1000 to avoid COID collisions with strategy-loop orders (0-indexed)
        coid = make_sleeve_coid(child.symbol, deps.now, 1000 + i)
        deps.broker.submit_simple_order(symbol=child.symbol, side=side_str, qty=qty,
                                        client_order_id=coid, dry_run=deps.dry_run)
        signed = qty if child.side is Side.BUY else -qty
        deps.ledger.record(Fill(symbol=child.symbol, qty=signed, price=price))
        mgr.record_fill(child.symbol, qty)
        submitted += 1
    return submitted


def recover_ledger(broker: Any, config: SleeveConfig) -> SleeveLedger:
    """Rebuild a SleeveLedger from the broker's current positions, restricted to the
    sleeve universe (daily-system holdings are ignored — they are a disjoint set)."""
    ledger = SleeveLedger()
    for p in broker.positions():
        if p.symbol in config.universe and p.qty != 0:
            ledger.record(Fill(symbol=p.symbol, qty=int(p.qty), price=float(p.avg_entry_price)))
    return ledger


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
    quotes: dict[str, QuoteBar] = {b.symbol: b for b in bars}
    ctx = _Ctx(deps.ledger, marks, quotes, deps.now)
    desired: list[Order] = []
    for b in bars:
        desired.extend(deps.strategy.on_event(b, ctx))

    # 5-7. Size, cap, reconcile, submit.
    n_orders = 0
    for order in desired:
        pos = deps.ledger.position(order.symbol)
        # An order that REDUCES an existing position is an exit. Exits must always
        # be allowed (de-risking) — they bypass the sleeve-room cap, which would
        # otherwise clamp them to 0 (the position is already inside gross_notional).
        is_reducing = pos != 0 and (
            (pos > 0 and order.side is Side.SELL) or (pos < 0 and order.side is Side.BUY)
        )
        has_prog = deps.exec_manager is not None and deps.exec_manager.has_active(order.symbol)
        is_new_open = pos == 0 and not has_prog
        # Trade-budget guard applies only to new opens, not to exits.
        if is_new_open and trade_budget_exhausted(round_trips=deps.ledger.round_trips, config=cfg):
            continue
        price = marks.get(order.symbol)
        if price is None:
            continue
        # Reducing order: cancel any active program first, then submit immediately.
        if is_reducing and deps.exec_manager is not None:
            deps.exec_manager.cancel(order.symbol)
        # New open with exec manager: route to manager (TWAP/A-C), don't submit now.
        if is_new_open and deps.exec_manager is not None:
            ec = deps.exec_config or ExecConfig()
            sigma, eta, gamma = calibrate(
                price=price,
                slice_shares=max(1, order.qty // ec.horizon_ticks),
                adv_dollar=_sleeve_adv_dollar(order.symbol, price),
                recent_returns=_recent_returns(deps, order.symbol),
                config=ec,
            )
            deps.exec_manager.start_entry(
                order, tick_index=deps.tick_index,
                sigma=sigma, eta=eta, gamma=gamma,
            )
            continue  # slices worked after the strategy loop (see below)
        # An in-flight program already owns this symbol's entry. The only permitted
        # action while it runs is an exit (is_reducing, handled above via cancel +
        # immediate submit). Any other order (re-fired entry, duplicate bar) must be
        # dropped so we never leak shares beyond the scheduled parent.
        # also catches pos==0 with a program already active (e.g. after crash recovery)
        if has_prog and not is_reducing:
            continue
        # Immediate path: exits, or new opens when no exec manager is present.
        if is_reducing:
            qty = min(order.qty, abs(pos))  # exits bypass caps; never over-close
        else:
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

    # Work in-flight execution programs (entry slices) AFTER the strategy step, so a
    # program created this tick gets its offset-0 slice now.
    n_orders += _work_active_programs(deps, marks, allocation)

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
        # deps_factory() is inside the guard too: a transient failure building deps
        # (e.g. fetching account equity) must not kill the daemon.
        try:
            deps = deps_factory()
            run_tick(deps)
            sleep_for = sleep_s if sleep_s is not None else deps.config.tick_seconds
        except Exception:
            logger.exception("intraday tick failed; continuing")
            sleep_for = sleep_s if sleep_s is not None else 60.0
        count += 1
        time.sleep(sleep_for)
