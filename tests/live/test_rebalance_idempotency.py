"""Rebalance must refuse to submit if same-day orders already exist (idempotency)."""

from __future__ import annotations

from datetime import date

from quant.live.rebalance import already_traded_today


class _ClientWithOrders:
    def list_orders_for_date(self, d: date) -> list[object]:
        return [object()]  # pretend one order already placed today


class _ClientNoOrders:
    def list_orders_for_date(self, d: date) -> list[object]:
        return []


def test_already_traded_today_true_when_orders_exist() -> None:
    assert already_traded_today(_ClientWithOrders(), date(2026, 6, 2)) is True


def test_already_traded_today_false_when_none() -> None:
    assert already_traded_today(_ClientNoOrders(), date(2026, 6, 2)) is False
