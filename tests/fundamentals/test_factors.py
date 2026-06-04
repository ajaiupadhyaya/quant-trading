"""Phase 7.B cross-sectional fundamentals read (value/quality on the mega-caps)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

import quant.fundamentals.factors as f
from quant.fundamentals.factors import (
    FundamentalRow,
    FundamentalsConfig,
    compute_fundamentals,
    live_fundamentals,
    render_fundamentals,
)

ASOF = date(2026, 6, 3)


def _row(sym: str, *, ey=None, btm=None, gp=None, ag=None) -> FundamentalRow:
    return FundamentalRow(
        symbol=sym,
        price=100.0,
        market_cap=1.0e12,
        earnings_yield=ey,
        book_to_market=btm,
        gross_profitability=gp,
        asset_growth=ag,
    )


def test_compute_labels_cheap_and_strong() -> None:
    rows = [_row(f"S{i}", ey=0.08, btm=0.5, gp=0.40, ag=0.05) for i in range(8)]
    r = compute_fundamentals(rows, asof=ASOF)
    assert r.coverage == 1.0 and r.n_covered == 8
    assert abs(r.median_earnings_yield - 0.08) < 1e-12
    assert r.valuation_label == "cheap"  # median E/P 8% >= 6%
    assert r.quality_label == "strong"  # median GP 0.40 >= 0.33


def test_compute_labels_rich_and_weak() -> None:
    rows = [_row(f"S{i}", ey=0.02, gp=0.10) for i in range(8)]
    r = compute_fundamentals(rows, asof=ASOF)
    assert r.valuation_label == "rich"  # 2% <= 3.5%
    assert r.quality_label == "weak"  # 0.10 <= 0.20


def test_compute_labels_fair_and_neutral() -> None:
    rows = [_row(f"S{i}", ey=0.045, gp=0.26) for i in range(8)]
    r = compute_fundamentals(rows, asof=ASOF)
    assert r.valuation_label == "fair"
    assert r.quality_label == "neutral"


def test_low_coverage_suppresses_labels_but_reports_medians() -> None:
    covered = [_row("A", ey=0.08, gp=0.40), _row("B", ey=0.07, gp=0.40)]
    uncovered = [_row(f"U{i}") for i in range(18)]
    r = compute_fundamentals(covered + uncovered, asof=ASOF)
    assert r.n_universe == 20 and r.n_covered == 2
    assert abs(r.coverage - 0.1) < 1e-12  # below the 0.4 floor
    assert r.median_earnings_yield is not None  # medians still computed from what exists
    assert r.valuation_label is None and r.quality_label is None


def test_empty_rows_never_raise() -> None:
    r = compute_fundamentals([], asof=ASOF)
    assert r.n_universe == 0 and r.n_covered == 0 and r.coverage == 0.0
    assert r.median_earnings_yield is None
    assert r.valuation_label is None and r.quality_label is None
    assert r.cheapest == () and r.richest == ()


def test_cheapest_and_richest_ranking() -> None:
    rows = [
        _row("A", ey=0.10),
        _row("B", ey=0.08),
        _row("C", ey=0.06),
        _row("D", ey=0.04),
        _row("E", ey=0.02),
    ]
    r = compute_fundamentals(rows, asof=ASOF, config=FundamentalsConfig(top_n=2))
    assert r.cheapest == ("A", "B")  # highest E/P
    assert r.richest == ("E", "D")  # lowest E/P, most-expensive-first


def test_median_ignores_none_and_nonfinite() -> None:
    rows = [_row("A", ey=0.06), _row("B", ey=None), _row("C", ey=float("nan"))]
    r = compute_fundamentals(rows, asof=ASOF)
    assert abs(r.median_earnings_yield - 0.06) < 1e-12  # only A counts


def test_render_unavailable_and_no_coverage() -> None:
    assert render_fundamentals(None) == "Fundamentals: unavailable"
    empty = compute_fundamentals([], asof=ASOF)
    assert render_fundamentals(empty) == "Fundamentals: no coverage"


def test_render_populated_contains_labels_and_metrics() -> None:
    rows = [_row(f"S{i}", ey=0.08, gp=0.40, ag=0.05) for i in range(8)]
    out = render_fundamentals(compute_fundamentals(rows, asof=ASOF))
    assert "valuation=cheap" in out
    assert "EY=8.0%" in out
    assert "quality=strong" in out
    assert "cov=100%" in out


def test_latest_prices_failopen(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant.data import bars

    monkeypatch.setattr(bars, "get_bars", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    out = f._latest_prices(SimpleNamespace(), ["AAPL", "MSFT"], ASOF, FundamentalsConfig())
    assert out == {}


def test_live_fundamentals_failopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prices down AND EDGAR down: read degrades to zero coverage, never raises."""
    from quant.data import edgar

    monkeypatch.setattr(f, "_latest_prices", lambda *a, **k: {})
    monkeypatch.setattr(
        edgar, "fetch_company_facts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    cfg = FundamentalsConfig(universe=("AAPL", "MSFT"))
    r = live_fundamentals(SimpleNamespace(), ASOF, config=cfg)
    assert r.n_universe == 2 and r.n_covered == 0 and r.coverage == 0.0
    assert r.valuation_label is None
