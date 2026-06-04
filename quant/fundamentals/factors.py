"""Cross-sectional fundamentals factors → a market-level valuation/quality read.

The trading book holds ETFs (no EDGAR facts) plus the multi-factor mega-cap
sleeve. EDGAR fundamentals only exist for operating companies, so we compute the
fundamentals read on the curated mega-cap universe and treat it as a *proxy* for
the broad equity market's posture — those 20 names are ~a third of SPY by weight.

Layering mirrors ``quant.macro.events``:
  * ``compute_fundamentals``  — PURE: per-symbol rows → aggregates + labels, no I/O.
  * ``fundamental_rows`` / ``live_fundamentals`` — bounded, fail-open I/O (cached
    prices + PIT EDGAR facts); never raise.
  * ``render_fundamentals``   — one terse line for the CLI / analyst prompt / logs.

Every per-symbol factor is PIT-correct (``quant.data.edgar`` filters facts by
``filed <= asof``). A name with missing facts contributes ``None`` and is simply
absent from the cross-sectional medians — the read degrades, it never breaks.
"""

from __future__ import annotations

import concurrent.futures
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from quant.data.universe import MEGACAP_UNIVERSE
from quant.util.logging import logger


@dataclass(frozen=True)
class FundamentalsConfig:
    """Universe + label anchors. Absolute anchors are advisory, not precise."""

    universe: tuple[str, ...] = tuple(MEGACAP_UNIVERSE)
    min_coverage: float = 0.4  # need >= this fraction resolved before labelling
    history_lookback_days: int = 21  # window of cached closes to source the latest price
    bars_timeout_s: float = 20.0
    top_n: int = 3  # names listed as cheapest / richest
    # Valuation label anchors on the median trailing earnings yield (net_income/mcap).
    cheap_ey: float = 0.06  # median E/P >= 6% → "cheap"
    rich_ey: float = 0.035  # median E/P <= 3.5% → "rich"
    # Quality label anchors on the median gross profitability (gross_profit/assets).
    strong_gp: float = 0.33
    weak_gp: float = 0.20


@dataclass(frozen=True)
class FundamentalRow:
    """One name's PIT factor read. Any field may be ``None`` (missing facts)."""

    symbol: str
    price: float | None
    market_cap: float | None
    earnings_yield: float | None  # net_income / market_cap (E/P)
    book_to_market: float | None  # stockholders_equity / market_cap
    gross_profitability: float | None  # gross_profit / total_assets (Novy-Marx)
    asset_growth: float | None  # raw YoY asset growth (investment factor, un-negated)


@dataclass(frozen=True)
class FundamentalsRead:
    """The market-level fundamentals posture for one ``asof`` date."""

    asof: str  # ISO
    n_universe: int
    n_covered: int
    coverage: float
    median_earnings_yield: float | None
    median_book_to_market: float | None
    median_gross_profitability: float | None
    median_asset_growth: float | None
    valuation_label: str | None  # "cheap" | "fair" | "rich"
    quality_label: str | None  # "strong" | "neutral" | "weak"
    cheapest: tuple[str, ...] = field(default_factory=tuple)  # highest-E/P names
    richest: tuple[str, ...] = field(default_factory=tuple)  # lowest-E/P names


def _finite(x: Any) -> float | None:
    try:
        v = float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and math.isfinite(v)) else None


def _median(vals: list[float | None]) -> float | None:
    clean = [v for v in (_finite(x) for x in vals) if v is not None]
    return statistics.median(clean) if clean else None


def compute_fundamentals(
    rows: list[FundamentalRow],
    *,
    asof: date,
    config: FundamentalsConfig | None = None,
) -> FundamentalsRead:
    """Pure: collapse per-symbol PIT rows into one market-level read.

    Labels are suppressed (``None``) when coverage is below ``min_coverage`` — a
    valuation call off two stragglers would be noise, not signal.
    """
    cfg = config or FundamentalsConfig()
    n_universe = len(rows)
    # "Covered" = market cap resolved AND at least one valuation factor present.
    covered = [r for r in rows if r.earnings_yield is not None or r.book_to_market is not None]
    n_covered = len(covered)
    coverage = (n_covered / n_universe) if n_universe else 0.0

    med_ey = _median([r.earnings_yield for r in rows])
    med_btm = _median([r.book_to_market for r in rows])
    med_gp = _median([r.gross_profitability for r in rows])
    med_ag = _median([r.asset_growth for r in rows])

    enough = coverage >= cfg.min_coverage
    valuation_label: str | None = None
    quality_label: str | None = None
    if enough and med_ey is not None:
        valuation_label = (
            "cheap" if med_ey >= cfg.cheap_ey else ("rich" if med_ey <= cfg.rich_ey else "fair")
        )
    if enough and med_gp is not None:
        quality_label = (
            "strong"
            if med_gp >= cfg.strong_gp
            else ("weak" if med_gp <= cfg.weak_gp else "neutral")
        )

    ranked = sorted(
        (r for r in rows if _finite(r.earnings_yield) is not None),
        key=lambda r: float(r.earnings_yield),  # type: ignore[arg-type]
        reverse=True,  # highest E/P (cheapest) first
    )
    cheapest = tuple(r.symbol for r in ranked[: cfg.top_n])
    # Lowest E/P (richest) — take the tail, present most-expensive-first.
    richest = tuple(r.symbol for r in reversed(ranked[-cfg.top_n :])) if ranked else ()

    return FundamentalsRead(
        asof=asof.isoformat(),
        n_universe=n_universe,
        n_covered=n_covered,
        coverage=coverage,
        median_earnings_yield=med_ey,
        median_book_to_market=med_btm,
        median_gross_profitability=med_gp,
        median_asset_growth=med_ag,
        valuation_label=valuation_label,
        quality_label=quality_label,
        cheapest=cheapest,
        richest=richest,
    )


def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def _latest_prices(
    settings: Any, symbols: list[str], asof: date, cfg: FundamentalsConfig
) -> dict[str, float]:
    """Latest cached close per symbol (filed <= asof). Bounded + fail-open → {}."""
    try:
        import pandas as pd

        from quant.data import bars
        from quant.strategies._common import field_frame

        start = asof - timedelta(days=cfg.history_lookback_days)
        req = bars.BarRequest(symbols=list(symbols), start=start, end=asof)
        frame = _with_timeout(lambda: bars.get_bars(req), cfg.bars_timeout_s)
        closes = field_frame(frame, "close")
        closes = closes.loc[: pd.Timestamp(asof)]  # hard PIT truncation
        out: dict[str, float] = {}
        for sym in symbols:
            if sym in closes.columns:
                s = closes[sym].dropna()
                if len(s):
                    px = _finite(s.iloc[-1])
                    if px is not None and px > 0:
                        out[sym] = px
        return out
    except Exception as exc:  # fail-open: no prices → market caps unresolved
        logger.info("fundamentals: price load skipped ({!r})", exc)
        return {}


def fundamental_rows(
    settings: Any, asof: date, *, config: FundamentalsConfig | None = None
) -> list[FundamentalRow]:
    """Build PIT factor rows for the configured universe. Never raises.

    Each EDGAR call is individually guarded — one bad ticker (no CIK, cache
    miss, malformed facts) yields ``None`` for the affected factor, not a crash.
    """
    cfg = config or FundamentalsConfig()
    symbols = list(cfg.universe)

    from quant.data.edgar import (
        asset_growth_yoy,
        book_to_market,
        earnings_yield,
        gross_profitability_ttm,
        market_cap_asof,
    )

    data_dir = getattr(settings, "data_dir", None)
    prices = _latest_prices(settings, symbols, asof, cfg)

    def _guard(value: Any) -> float | None:
        # Each EDGAR call is wrapped at the call site; this just coerces+sanitizes.
        return _finite(value)

    rows: list[FundamentalRow] = []
    for sym in symbols:
        price = prices.get(sym)
        mcap = ey = btm = gp = ag = None
        if price is not None and price > 0:
            try:
                mcap = _guard(market_cap_asof(sym, asof, price=price, data_dir=data_dir))
            except Exception:  # no CIK / cache miss / malformed facts → factor absent
                mcap = None
        if mcap is not None and mcap > 0:
            try:
                ey = _guard(earnings_yield(sym, asof, market_cap=mcap, data_dir=data_dir))
            except Exception:
                ey = None
            try:
                btm = _guard(book_to_market(sym, asof, market_cap=mcap, data_dir=data_dir))
            except Exception:
                btm = None
        try:
            gp = _guard(gross_profitability_ttm(sym, asof, data_dir=data_dir))
        except Exception:
            gp = None
        try:
            ag = _guard(asset_growth_yoy(sym, asof, data_dir=data_dir))
        except Exception:
            ag = None
        rows.append(
            FundamentalRow(
                symbol=sym,
                price=price,
                market_cap=mcap,
                earnings_yield=ey,
                book_to_market=btm,
                gross_profitability=gp,
                asset_growth=ag,
            )
        )
    return rows


def live_fundamentals(
    settings: Any, asof: date, *, config: FundamentalsConfig | None = None
) -> FundamentalsRead:
    """Bounded, fail-open fundamentals read for ``asof``. Never raises."""
    cfg = config or FundamentalsConfig()
    rows = fundamental_rows(settings, asof, config=cfg)
    return compute_fundamentals(rows, asof=asof, config=cfg)


def render_fundamentals(r: FundamentalsRead | None) -> str:
    """Terse one-liner for the Claude prompt + CLI + logs."""
    if r is None:
        return "Fundamentals: unavailable"
    if r.n_covered == 0:
        return "Fundamentals: no coverage"
    bits: list[str] = []
    if r.valuation_label:
        bits.append(f"valuation={r.valuation_label}")
    if r.median_earnings_yield is not None:
        bits.append(f"EY={r.median_earnings_yield:.1%}")
    if r.quality_label:
        bits.append(f"quality={r.quality_label}")
    if r.median_gross_profitability is not None:
        bits.append(f"GP={r.median_gross_profitability:.2f}")
    if r.median_asset_growth is not None:
        bits.append(f"asset_growth={r.median_asset_growth:+.1%}")
    bits.append(f"cov={r.coverage:.0%}({r.n_covered}/{r.n_universe})")
    if r.cheapest:
        bits.append(f"cheap={'/'.join(r.cheapest)}")
    return "Fundamentals: " + ", ".join(bits)
