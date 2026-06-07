from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.guardrails import (
    clamp_qty_to_caps,
    daily_loss_breached,
    in_flat_window,
    trade_budget_exhausted,
)


def test_clamp_to_per_trade_cap():
    c = SleeveConfig(per_trade_cap=2_000.0)
    qty = clamp_qty_to_caps(desired_qty=100, price=100.0, gross_notional=0.0,
                            sleeve_allocation=10_000.0, config=c)
    assert qty == 20


def test_clamp_to_remaining_sleeve_room():
    c = SleeveConfig(per_trade_cap=5_000.0)
    qty = clamp_qty_to_caps(desired_qty=40, price=100.0, gross_notional=9_000.0,
                            sleeve_allocation=10_000.0, config=c)
    assert qty == 10


def test_clamp_never_negative():
    c = SleeveConfig()
    qty = clamp_qty_to_caps(desired_qty=10, price=100.0, gross_notional=10_000.0,
                            sleeve_allocation=10_000.0, config=c)
    assert qty == 0


def test_trade_budget_exhausted():
    c = SleeveConfig(max_round_trips=20)
    assert trade_budget_exhausted(round_trips=20, config=c) is True
    assert trade_budget_exhausted(round_trips=19, config=c) is False


def test_daily_loss_breached():
    c = SleeveConfig(daily_loss_halt_pct=0.015)
    assert daily_loss_breached(day_pnl=-150.0, sleeve_allocation=10_000.0, config=c) is True
    assert daily_loss_breached(day_pnl=-149.0, sleeve_allocation=10_000.0, config=c) is False
    assert daily_loss_breached(day_pnl=500.0, sleeve_allocation=10_000.0, config=c) is False


def test_in_flat_window():
    c = SleeveConfig(flat_by_close_minutes=15)
    close = datetime(2026, 6, 8, 20, 0, tzinfo=UTC)  # 16:00 ET == 20:00 UTC
    assert in_flat_window(datetime(2026, 6, 8, 19, 46, tzinfo=UTC), close, c) is True
    assert in_flat_window(datetime(2026, 6, 8, 19, 44, tzinfo=UTC), close, c) is False


@given(
    desired=st.integers(min_value=0, max_value=100_000),
    price=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
    gross=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False),
)
def test_property_clamp_never_exceeds_caps(desired, price, gross):
    """Spec invariant: a clamped order NEVER breaches the per-trade cap nor the
    remaining sleeve room, for ANY input."""
    c = SleeveConfig()
    qty = clamp_qty_to_caps(desired_qty=desired, price=price, gross_notional=gross,
                            sleeve_allocation=10_000.0, config=c)
    assert qty >= 0
    assert qty * price <= c.per_trade_cap + price          # within one share of cap
    assert qty * price <= max(0.0, 10_000.0 - gross) + price
    assert qty <= desired
