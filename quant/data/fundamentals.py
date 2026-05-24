"""Fundamentals stub backed by yfinance.

NOTE: this is a Plan 1 (Foundation) stub. The multi-factor strategy in Plan 4
will replace yfinance with SEC EDGAR point-in-time data to avoid look-ahead
bias. Until then, this is fine for sanity-checking the rest of the plumbing.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from quant.util.logging import logger


def get_fundamentals(symbol: str) -> dict[str, Any]:
    """Return yfinance's `info` dict for the symbol.

    Returns an empty dict on failure rather than raising — callers should
    handle missing fields with `dict.get(..., default)`.
    """
    try:
        return dict(yf.Ticker(symbol).info or {})
    except Exception as exc:
        logger.warning("yfinance fundamentals failed for {}: {}", symbol, exc)
        return {}


def book_to_market(symbol: str) -> float:
    """Book-to-market ratio, derived as 1 / priceToBook."""
    info = get_fundamentals(symbol)
    pb = info.get("priceToBook")
    if pb is None or pb == 0:
        return float("nan")
    return 1.0 / float(pb)


def gross_profitability(symbol: str) -> float:
    """Gross profitability = grossProfits / totalAssets (Novy-Marx 2013)."""
    info = get_fundamentals(symbol)
    gp = info.get("grossProfits")
    assets = info.get("totalAssets")
    if gp is None or assets is None or assets == 0:
        return float("nan")
    return float(gp) / float(assets)
