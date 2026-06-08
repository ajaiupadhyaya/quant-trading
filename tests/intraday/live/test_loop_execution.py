from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.execution.config import ExecConfig
from quant.intraday.execution.manager import ExecutionManager
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.loop import TickDeps, run_tick
from quant.intraday.live.sleeve import SleeveLedger
from quant.intraday.strategy import Order, Side


class _Broker:
    def __init__(self):
        self.orders = []
    def account(self):
        class A:
            equity = 100_000.0
        return A()
    def submit_simple_order(self, *, symbol, side, qty, client_order_id,
                            order_type="market", limit_price=None, dry_run=False):
        self.orders.append((symbol, side, qty))
        return client_order_id


class _Feed:
    def __init__(self, bars): self._bars = bars
    def latest_quotes(self, now=None): return self._bars


class _Strat:
    def __init__(self, orders): self._orders = orders
    def on_event(self, event, ctx): return self._orders


def _qb(sym, price):
    return QuoteBar(ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC), symbol=sym,
                    bid=price - 0.01, ask=price + 0.01, bid_size=100, ask_size=100)


def _deps(tmp_path, broker, feed, strat, ledger, *, tick_index, mgr, now=None):
    return TickDeps(
        data_dir=tmp_path,
        config=SleeveConfig(notional_cap_pct=1.0, notional_cap_abs=1e9,
                            per_trade_cap=1e9, mean_reversion_lookback=5),
        broker=broker, feed=feed, strategy=strat, ledger=ledger,
        now=now or datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        session_open=True, session_close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
        tick_index=tick_index, exec_manager=mgr,
    )


def test_entry_is_worked_over_ticks_not_dumped(tmp_path):
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3, risk_aversion=1e-12))
    broker, ledger = _Broker(), SleeveLedger()
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                   _Strat([Order("QQQ", Side.BUY, 90)]), ledger, tick_index=0, mgr=mgr))
    first = sum(q for s, _, q in broker.orders if s == "QQQ")
    assert first <= 30  # near-TWAP over 3 ticks → ~30 first slice; proves we sliced, not dumped
    silent = _Strat([])
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), silent, ledger,
                   tick_index=1, mgr=mgr))
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), silent, ledger,
                   tick_index=2, mgr=mgr))
    assert sum(q for s, _, q in broker.orders if s == "QQQ") == 90  # fully worked


def test_refired_entry_while_program_active_does_not_leak(tmp_path):
    # A persistent BUY signal must NOT add shares beyond the scheduled parent while
    # the program is still working.
    mgr = ExecutionManager(ExecConfig(horizon_ticks=3, risk_aversion=1e-12))
    broker, ledger = _Broker(), SleeveLedger()
    persistent = _Strat([Order("QQQ", Side.BUY, 90)])  # fires every tick
    for t in range(3):
        run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                       persistent, ledger, tick_index=t, mgr=mgr))
    # Exactly the scheduled parent (90) is bought across the 3 ticks — no leak.
    assert sum(q for s, side, q in broker.orders if s == "QQQ" and side == "buy") == 90


def test_flatten_cancels_active_program(tmp_path):
    mgr = ExecutionManager(ExecConfig(horizon_ticks=5, risk_aversion=1e-12))
    broker, ledger = _Broker(), SleeveLedger()
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]),
                   _Strat([Order("QQQ", Side.BUY, 100)]), ledger, tick_index=0, mgr=mgr))
    assert mgr.has_active("QQQ")
    run_tick(_deps(tmp_path, broker, _Feed([_qb("QQQ", 100.0)]), _Strat([]), ledger,
                   tick_index=1, mgr=mgr, now=datetime(2026, 6, 8, 19, 50, tzinfo=UTC)))
    assert not mgr.has_active("QQQ")
    assert any(s == "QQQ" and side == "sell" for s, side, _ in broker.orders)
