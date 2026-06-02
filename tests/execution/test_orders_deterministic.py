"""client_order_id must be deterministic per (strategy, symbol, session-date)
so the broker rejects a duplicate same-day resubmission (idempotency)."""

from __future__ import annotations

from datetime import date

from quant.execution.orders import make_client_order_id


def test_client_order_id_is_deterministic() -> None:
    a = make_client_order_id("trend", "SPY", date(2026, 6, 2))
    b = make_client_order_id("trend", "SPY", date(2026, 6, 2))
    assert a == b


def test_client_order_id_differs_by_symbol_and_date() -> None:
    assert make_client_order_id("trend", "SPY", date(2026, 6, 2)) != make_client_order_id(
        "trend", "EFA", date(2026, 6, 2)
    )
    assert make_client_order_id("trend", "SPY", date(2026, 6, 2)) != make_client_order_id(
        "trend", "SPY", date(2026, 6, 3)
    )


def test_client_order_id_carries_slug_prefix_for_attribution() -> None:
    coid = make_client_order_id("multi-factor", "JPM", date(2026, 6, 2))
    assert coid.startswith("multi-factor-20260602-JPM")
    assert len(coid) <= 48  # Alpaca client_order_id length limit
