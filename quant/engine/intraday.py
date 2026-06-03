"""Intraday snapshot signals (Phase 7A) — make the engine react WITHIN the session.

The daily signals battery (quant.research.signals) refreshes once pre-open; this
module adds live INTRADAY reactivity from a single bounded Alpaca snapshot call
per cycle: for each ETF it reads the latest price, today's OHLC-so-far, and the
prior close, and derives the day's return, market breadth, a Parkinson range-vol
proxy, and cross-sectional dispersion. Pure compute + a fail-open fetcher; like
everything in the engine it is READ-ONLY and places no orders.
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import Any

from quant.data.universe import ETF_UNIVERSE
from quant.util.logging import logger

_MARKET = "SPY"
_PARKINSON_K = 2.0 * math.sqrt(math.log(2.0))  # sigma_day = ln(H/L) / (2*sqrt(ln2))
_ANNUALIZE = math.sqrt(252.0)


@dataclass(frozen=True)
class IntradaySignals:
    """One cycle's intraday read of the ETF universe (None when unavailable)."""

    asof_minute: str | None  # latest minute-bar timestamp (freshness)
    spy_ret: float | None  # SPY return today vs prior close
    breadth: float | None  # fraction of the universe up on the day
    range_vol: float | None  # SPY Parkinson annualized vol proxy (today H/L)
    dispersion: float | None  # cross-sectional std of intraday returns
    n_symbols: int
    n_up: int
    n_down: int


_EMPTY = IntradaySignals(
    asof_minute=None,
    spy_ret=None,
    breadth=None,
    range_vol=None,
    dispersion=None,
    n_symbols=0,
    n_up=0,
    n_down=0,
)


def _finite(x: Any) -> float | None:
    try:
        v = float(x) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if (v is not None and math.isfinite(v)) else None


def _with_timeout(fn: Any, seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=seconds)


def fetch_intraday_snapshot(
    settings: Any, symbols: list[str] | None = None, *, timeout_s: float = 10.0
) -> dict[str, dict[str, float | None]] | None:
    """One bounded Alpaca snapshot call. Returns ``{sym: {price, prev_close, high,
    low, minute_ts}}`` or ``None`` on any failure (fail-open)."""
    syms = symbols if symbols is not None else list(ETF_UNIVERSE)
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest

        client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key
        )
        snaps = _with_timeout(
            lambda: client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=syms)),
            timeout_s,
        )
    except Exception as exc:  # fail-open: no intraday this cycle
        logger.info("engine.intraday: snapshot skipped ({!r})", exc)
        return None

    out: dict[str, dict[str, float | None]] = {}
    for sym, snap in (snaps or {}).items():
        try:
            mb = getattr(snap, "minute_bar", None)
            db = getattr(snap, "daily_bar", None)
            pdb = getattr(snap, "previous_daily_bar", None)
            lt = getattr(snap, "latest_trade", None)
            price = (
                _finite(getattr(mb, "close", None))
                or _finite(getattr(db, "close", None))
                or _finite(getattr(lt, "price", None))
            )
            ts = getattr(mb, "timestamp", None) or getattr(db, "timestamp", None)
            out[str(sym)] = {
                "price": price,
                "prev_close": _finite(getattr(pdb, "close", None)),
                "high": _finite(getattr(db, "high", None)),
                "low": _finite(getattr(db, "low", None)),
                "minute_ts": ts.isoformat() if ts is not None else None,
            }
        except Exception:  # one bad symbol must not sink the snapshot
            continue
    return out or None


def compute_intraday_signals(
    snap: dict[str, dict[str, Any]] | None, *, market: str = _MARKET
) -> IntradaySignals:
    """Pure: derive intraday aggregates from a snapshot dict. Never raises."""
    if not snap:
        return _EMPTY

    rets: dict[str, float] = {}
    for sym, row in snap.items():
        price = _finite(row.get("price"))
        prev = _finite(row.get("prev_close"))
        if price is not None and prev is not None and prev > 0:
            rets[sym] = price / prev - 1.0

    n = len(rets)
    n_up = sum(1 for r in rets.values() if r > 0)
    n_down = sum(1 for r in rets.values() if r < 0)
    breadth = (n_up / n) if n else None
    dispersion = None
    if n >= 2:
        vals = list(rets.values())
        mean = sum(vals) / n
        dispersion = _finite(math.sqrt(sum((v - mean) ** 2 for v in vals) / n))

    # SPY Parkinson range-vol proxy from today's developing high/low.
    range_vol = None
    mrow = snap.get(market, {})
    high, low = _finite(mrow.get("high")), _finite(mrow.get("low"))
    if high is not None and low is not None and low > 0 and high >= low:
        rng = math.log(high / low)
        range_vol = _finite((rng / _PARKINSON_K) * _ANNUALIZE)

    asof: str | None = None
    if market in snap:
        mts = snap[market].get("minute_ts")
        asof = str(mts) if mts is not None else None
    if asof is None:
        ts_vals = [str(r.get("minute_ts")) for r in snap.values() if r.get("minute_ts")]
        asof = max(ts_vals) if ts_vals else None

    return IntradaySignals(
        asof_minute=asof,
        spy_ret=rets.get(market),
        breadth=breadth,
        range_vol=range_vol,
        dispersion=dispersion,
        n_symbols=n,
        n_up=n_up,
        n_down=n_down,
    )


def live_intraday_signals(settings: Any, symbols: list[str] | None = None) -> IntradaySignals:
    """Fetch + compute in one fail-open call (the loop's default seam)."""
    return compute_intraday_signals(fetch_intraday_snapshot(settings, symbols))
