import dataclasses
from datetime import UTC, datetime

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.loop import TickDeps, run_loop, run_tick
from quant.intraday.live.sleeve import Fill, SleeveLedger


class _Broker:
    def __init__(self):
        self.orders = []

    def account(self):
        class A:
            equity = 100_000.0

        return A()

    def submit_simple_order(
        self,
        *,
        symbol,
        side,
        qty,
        client_order_id,
        order_type="market",
        limit_price=None,
        dry_run=False,
    ):
        self.orders.append((symbol, side, qty))
        return client_order_id


class _Feed:
    def __init__(self, bars):
        self._bars = bars

    def latest_quotes(self, now=None):
        return self._bars


class _Strat:
    """Stub strategy: emits a fixed order list regardless of input."""

    def __init__(self, orders):
        self._orders = orders

    def on_event(self, event, ctx):
        return self._orders


def _deps(tmp_path, broker, feed, strat, ledger, now, close):
    return TickDeps(
        data_dir=tmp_path,
        config=SleeveConfig(),
        broker=broker,
        feed=feed,
        strategy=strat,
        ledger=ledger,
        now=now,
        session_open=True,
        session_close=close,
    )


def _qb(sym, price):
    return QuoteBar(
        ts=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        symbol=sym,
        bid=price - 0.01,
        ask=price + 0.01,
        bid_size=100,
        ask_size=100,
    )


def test_session_closed_does_nothing(tmp_path):
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    deps = _deps(
        tmp_path,
        broker,
        _Feed([_qb("QQQ", 100.0)]),
        _Strat([Order("QQQ", Side.BUY, 10)]),
        ledger,
        now=datetime(2026, 6, 8, 9, 0, tzinfo=UTC),
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    deps = dataclasses.replace(deps, session_open=False)
    run_tick(deps)
    assert broker.orders == []


def test_happy_path_submits_capped_order(tmp_path):
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    deps = _deps(
        tmp_path,
        broker,
        _Feed([_qb("QQQ", 100.0)]),
        _Strat([Order("QQQ", Side.BUY, 1000)]),
        ledger,
        now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert len(broker.orders) == 1
    sym, side, qty = broker.orders[0]
    assert sym == "QQQ" and side == "buy" and qty == 20
    assert ledger.position("QQQ") == 20


def test_daily_loss_breach_flattens_and_halts(tmp_path):
    from quant.intraday.live.halt import load_sleeve_halt
    from quant.intraday.live.sleeve import Fill
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    ledger.record(Fill("QQQ", 100, 100.0))
    deps = _deps(
        tmp_path,
        broker,
        _Feed([_qb("QQQ", 50.0)]),
        _Strat([Order("QQQ", Side.BUY, 10)]),
        ledger,
        now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert ("QQQ", "sell", 100) in broker.orders
    assert len(broker.orders) == 1  # the strategy's BUY entry was NOT placed
    assert ledger.position("QQQ") == 0
    assert load_sleeve_halt(tmp_path).active is True


def test_flat_by_close_flattens_and_skips_entries(tmp_path):
    from quant.intraday.live.sleeve import Fill
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    ledger.record(Fill("QQQ", 10, 100.0))
    deps = _deps(
        tmp_path,
        broker,
        _Feed([_qb("QQQ", 100.0)]),
        _Strat([Order("QQQ", Side.BUY, 10)]),
        ledger,
        now=datetime(2026, 6, 8, 19, 50, tzinfo=UTC),
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert ("QQQ", "sell", 10) in broker.orders
    assert len(broker.orders) == 1  # only the flatten; no new entry
    assert ledger.position("QQQ") == 0


def test_global_halt_blocks_all_orders(tmp_path):
    from quant.governance.halt import set_halt
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    set_halt(tmp_path, reason="global stop")
    deps = _deps(
        tmp_path,
        broker,
        _Feed([_qb("QQQ", 100.0)]),
        _Strat([Order("QQQ", Side.BUY, 10)]),
        ledger,
        now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert broker.orders == []


def test_feed_error_skips_tick(tmp_path):
    from quant.intraday.live.feed import FeedError
    from quant.intraday.strategy import Order, Side

    class _BadFeed:
        def latest_quotes(self, now=None):
            raise FeedError("network down")

    broker, ledger = _Broker(), SleeveLedger()
    deps = _deps(
        tmp_path,
        broker,
        _BadFeed(),
        _Strat([Order("QQQ", Side.BUY, 10)]),
        ledger,
        now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)  # must not raise
    assert broker.orders == []


def test_trade_budget_blocks_new_opens(tmp_path):
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    ledger.record(Fill("IWM", 5, 50.0))  # one open -> round_trips == 1
    deps = TickDeps(
        data_dir=tmp_path,
        config=SleeveConfig(max_round_trips=1),
        broker=broker,
        feed=_Feed([_qb("QQQ", 100.0)]),
        strategy=_Strat([Order("QQQ", Side.BUY, 10)]),
        ledger=ledger,
        now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        session_open=True,
        session_close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert all(sym != "QQQ" for sym, _, _ in broker.orders)  # new open was blocked


def test_strategy_exit_executes_when_sleeve_full(tmp_path):
    """CRITICAL: a strategy exit must bypass the sleeve-room cap and close the
    position even when the sleeve is at/over capacity."""
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    ledger.record(Fill("QQQ", 20, 100.0))  # long 20; gross ~ 2000
    deps = TickDeps(
        data_dir=tmp_path,
        config=SleeveConfig(notional_cap_pct=0.0001, notional_cap_abs=10.0),
        broker=broker,
        feed=_Feed([_qb("QQQ", 100.0)]),
        strategy=_Strat([Order("QQQ", Side.SELL, 20)]),  # exit
        ledger=ledger,
        now=datetime(2026, 6, 8, 15, 0, tzinfo=UTC),
        session_open=True,
        session_close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert ("QQQ", "sell", 20) in broker.orders
    assert ledger.position("QQQ") == 0


def test_short_flatten_buys_to_cover(tmp_path):
    from quant.intraday.strategy import Order, Side

    broker, ledger = _Broker(), SleeveLedger()
    ledger.record(Fill("QQQ", -10, 100.0))  # short 10
    deps = _deps(
        tmp_path,
        broker,
        _Feed([_qb("QQQ", 100.0)]),
        _Strat([Order("QQQ", Side.SELL, 10)]),
        ledger,
        now=datetime(2026, 6, 8, 19, 50, tzinfo=UTC),  # flat-by-close window
        close=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
    )
    run_tick(deps)
    assert ("QQQ", "buy", 10) in broker.orders
    assert ledger.position("QQQ") == 0


def test_run_loop_survives_bad_tick(tmp_path):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        raise RuntimeError("transient deps failure")

    # max_ticks=2, sleep_s=0 -> must complete without raising despite factory errors
    run_loop(factory, max_ticks=2, sleep_s=0)
    assert calls["n"] == 2
