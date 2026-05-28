"""Tests for portfolio pre-trade risk reporting."""

from __future__ import annotations

from quant.execution.orders import OrderSide, OrderTemplate
from quant.risk.pretrade import RiskLimits, build_pretrade_report


def test_pretrade_report_blocks_concentration_and_gross_exposure() -> None:
    report = build_pretrade_report(
        equity=100_000.0,
        orders=[
            OrderTemplate("SPY", 300, OrderSide.BUY, "baseline"),
            OrderTemplate("TLT", 300, OrderSide.BUY, "baseline"),
        ],
        reference_prices={"SPY": 500.0, "TLT": 100.0},
        limits=RiskLimits(max_gross_exposure=1.0, max_symbol_weight=0.40),
    )

    assert not report.passed
    assert any(v.code == "gross_exposure" for v in report.violations)
    assert any(v.code == "symbol_concentration" and v.symbol == "SPY" for v in report.violations)


def test_pretrade_report_passes_diversified_order_plan() -> None:
    report = build_pretrade_report(
        equity=100_000.0,
        orders=[
            OrderTemplate("SPY", 40, OrderSide.BUY, "baseline"),
            OrderTemplate("TLT", 100, OrderSide.BUY, "baseline"),
        ],
        reference_prices={"SPY": 500.0, "TLT": 100.0},
        limits=RiskLimits(max_gross_exposure=1.0, max_symbol_weight=0.40),
    )

    assert report.passed
    assert report.gross_exposure == 0.30
