"""End-to-end integration test for the intraday tick loop.

Drives many ticks through the REAL run_tick() with:
  - fake clock (scripted datetime sequence)
  - fake feed (scripted price path via price_fn)
  - fake broker (records submitted orders)
  - REAL MeanReversionStrategy
  - REAL SleeveLedger, REAL guardrails, REAL halt mechanics

Invariants verified:
  (a) Per-trade cap is respected on every submitted order.
  (b) Flat-by-close empties all positions.
  (c) Daily-loss halt fires and freezes subsequent trading.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant.intraday.data.events import QuoteBar
from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.halt import load_sleeve_halt
from quant.intraday.live.loop import TickDeps, run_tick
from quant.intraday.live.sleeve import Fill, SleeveLedger
from quant.intraday.live.strategy import MeanReversionStrategy

# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class _Broker:
    """Records (symbol, side, qty) for every submitted order."""

    def __init__(self) -> None:
        self.orders: list[tuple[str, str, int]] = []

    def account(self) -> object:
        class _A:
            equity: float = 100_000.0

        return _A()

    def positions(self) -> list[object]:
        return []

    def submit_simple_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        client_order_id: str,
        order_type: str = "market",
        limit_price: float | None = None,
        dry_run: bool = False,
    ) -> str:
        self.orders.append((symbol, side, qty))
        return client_order_id


class _Feed:
    """Returns a single QQQ QuoteBar per call, price derived from price_fn(now)."""

    def __init__(self, price_fn: Callable[[datetime], float]) -> None:
        self._price_fn = price_fn

    def latest_quotes(self, now: datetime | None = None) -> list[QuoteBar]:
        ts = now or datetime.now(UTC)
        price = self._price_fn(ts)
        return [
            QuoteBar(
                ts=ts,
                symbol="QQQ",
                bid=round(price - 0.01, 4),
                ask=round(price + 0.01, 4),
                bid_size=500,
                ask_size=500,
            )
        ]


# ---------------------------------------------------------------------------
# Helper to build TickDeps
# ---------------------------------------------------------------------------


def _make_deps(
    tmp_path: Path,
    broker: _Broker,
    feed: _Feed,
    strategy: MeanReversionStrategy,
    ledger: SleeveLedger,
    cfg: SleeveConfig,
    now: datetime,
    session_close: datetime,
) -> TickDeps:
    return TickDeps(
        data_dir=tmp_path,
        config=cfg,
        broker=broker,
        feed=feed,
        strategy=strategy,
        ledger=ledger,
        now=now,
        session_open=True,
        session_close=session_close,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# Test 1: per-trade cap is respected, and flat-by-close empties positions
# ---------------------------------------------------------------------------


def test_caps_respected_and_flat_by_close(tmp_path: Path) -> None:
    """Drive a scripted price path through the REAL MeanReversionStrategy.

    Price path design (lookback=5):
      Ticks 0-4  : price = 100.0  (flat -- fills the rolling window)
      Tick 5     : price = 103.0  (spike up; prior window is full + constant →
                                   sd=0 → z=+1e9 → SELL signal fires)
      Tick 6 (close window): price = 100.0, now is within flat_by_close_minutes
                             → _flatten_all should close any open position.

    Cap arithmetic:
      per_trade_cap=2_000.0, price≈100.0  →  floor(2000/100) = 20 shares max.
      unit_shares=1000 deliberately exceeds the cap so the clamp must bite.

    Assertions:
      (a) At least one order was placed (strategy fires).
      (b) Every placed order has qty ≤ 20 (per-trade cap = 2000 / ~100).
      (c) After the flat-by-close tick, ledger.positions() == {} (fully flat).
    """
    cfg = SleeveConfig(
        per_trade_cap=2_000.0,
        mean_reversion_lookback=5,
        flat_by_close_minutes=15,
        daily_loss_halt_pct=0.015,  # default, not triggered here
    )
    strat = MeanReversionStrategy(cfg, unit_shares=1000)  # huge so cap must bite
    broker = _Broker()
    ledger = SleeveLedger()

    # Session: open at 09:30, close at 16:00 (UTC+0 for simplicity)
    session_close = datetime(2026, 6, 9, 16, 0, tzinfo=UTC)

    # Build the scripted price sequence.
    # Ticks 0-4: 100.0 (warm-up window), tick 5: 103.0 (spike triggers entry).
    base_time = datetime(2026, 6, 9, 14, 0, tzinfo=UTC)  # well before close
    prices = [100.0] * 5 + [103.0]

    # --- Phase 1: drive the warm-up + signal ticks ---
    for i, price in enumerate(prices):
        now = base_time + timedelta(minutes=i)

        def _price_fn(t: datetime, p: float = price) -> float:
            return p

        feed = _Feed(_price_fn)
        deps = _make_deps(tmp_path, broker, feed, strat, ledger, cfg, now, session_close)
        run_tick(deps)

    # (a) At least one order was placed by the real strategy.
    assert len(broker.orders) >= 1, (
        "Expected at least one order from the mean-reversion strategy; none was placed. "
        "Check that the price spike at tick 5 produces a signal with the given lookback."
    )

    # (b) Every placed order must respect the per-trade cap.
    max_allowed_qty = int(cfg.per_trade_cap // 100.0)  # = 20
    for sym, side, qty in broker.orders:
        assert qty <= max_allowed_qty, (
            f"Order qty {qty} for {sym}/{side} exceeds per-trade cap "
            f"({max_allowed_qty} shares at ~$100). Cap logic is broken."
        )

    # --- Phase 2: flat-by-close tick ---
    # Move 'now' into the flat-by-close window (within flat_by_close_minutes of close).
    flat_now = session_close - timedelta(minutes=cfg.flat_by_close_minutes - 1)

    flat_price = 100.0

    def _flat_price_fn(t: datetime) -> float:
        return flat_price

    flat_feed = _Feed(_flat_price_fn)
    flat_deps = _make_deps(tmp_path, broker, flat_feed, strat, ledger, cfg, flat_now, session_close)
    run_tick(flat_deps)

    # (c) All positions must be closed after flat-by-close.
    assert ledger.positions() == {}, (
        f"Expected no open positions after flat-by-close tick; got {ledger.positions()}"
    )


# ---------------------------------------------------------------------------
# Test 2: daily-loss halt fires and freezes subsequent trading
# ---------------------------------------------------------------------------


def test_loss_halt_stops_subsequent_trading(tmp_path: Path) -> None:
    """Seed a long position then mark it down so day_pnl breaches the loss threshold.

    Design:
      - equity=100_000, notional_cap_pct=0.10 → sleeve_allocation=10_000
      - daily_loss_halt_pct=0.01 → threshold = -100.0
      - Seed long 100 shares @ 100.0 (notional=10_000, avg_cost=100.0).
      - Drop price to 98.9 → unrealized = (98.9 - 100.0) * 100 = -110 < -100 → breach.

    Assertions:
      (d) After the breaching tick, load_sleeve_halt(tmp_path).active is True.
      (e) Running one more tick does NOT increase len(broker.orders)
          (the loop exits early on the halt check before doing anything).
    """
    cfg = SleeveConfig(
        mean_reversion_lookback=5,
        daily_loss_halt_pct=0.01,  # 1% of sleeve_allocation = $100 threshold
        notional_cap_pct=0.10,
        notional_cap_abs=10_000.0,
        per_trade_cap=2_000.0,
    )
    # We use a fresh strategy here; it will not fire because the halt fires first.
    strat = MeanReversionStrategy(cfg, unit_shares=10)
    broker = _Broker()
    ledger = SleeveLedger()

    # Seed a long position: 100 shares @ $100.0 → avg_cost=100.0
    ledger.record(Fill(symbol="QQQ", qty=100, price=100.0))

    session_close = datetime(2026, 6, 9, 16, 0, tzinfo=UTC)
    # Now is well before close (no flat-by-close interference).
    tick_now = datetime(2026, 6, 9, 14, 30, tzinfo=UTC)

    # Price that triggers the loss threshold:
    # sleeve_allocation = min(100_000 * 0.10, 10_000) = 10_000
    # threshold = -0.01 * 10_000 = -100.0
    # unrealized @ 98.9 = (98.9 - 100.0) * 100 = -110 < -100  ✓
    crash_price = 98.9

    def _crash_price_fn(t: datetime) -> float:
        return crash_price

    crash_feed = _Feed(_crash_price_fn)
    deps = _make_deps(tmp_path, broker, crash_feed, strat, ledger, cfg, tick_now, session_close)
    run_tick(deps)

    # (d) Sleeve halt must be active.
    halt_state = load_sleeve_halt(tmp_path)
    assert halt_state.active is True, (
        f"Expected sleeve halt to be active after loss breach; got active={halt_state.active}, "
        f"reason={halt_state.reason!r}"
    )

    # Record order count right after the breaching tick.
    orders_after_breach = len(broker.orders)

    # (e) Run a second tick — the halt check must fire before the feed is called /
    #     any order is submitted, so order count must not grow.
    strat2 = MeanReversionStrategy(cfg, unit_shares=10)
    subsequent_deps = _make_deps(
        tmp_path,
        broker,
        crash_feed,
        strat2,
        ledger,
        cfg,
        tick_now + timedelta(minutes=1),
        session_close,
    )
    run_tick(subsequent_deps)

    assert len(broker.orders) == orders_after_breach, (
        f"Expected no new orders after halt, but {len(broker.orders) - orders_after_breach} "
        "extra order(s) were submitted. Halt check is not firing early enough."
    )
