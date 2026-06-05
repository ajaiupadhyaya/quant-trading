"""Live fundamentals read (roadmap Phase 7.B).

A cross-sectional value/quality/investment snapshot of the mega-cap operating
universe — the only names in our book with SEC EDGAR fundamentals — used as an
advisory proxy for the broad equity market's valuation + quality posture. Every
read is PIT-correct (reuses ``quant.data.edgar``), cached, fail-open, and
feeds the MarketState + analyst context as advisory only.
"""

from __future__ import annotations

from quant.fundamentals.factors import (
    FundamentalRow,
    FundamentalsConfig,
    FundamentalsRead,
    compute_fundamentals,
    fundamental_rows,
    live_fundamentals,
    render_fundamentals,
)

__all__ = [
    "FundamentalRow",
    "FundamentalsConfig",
    "FundamentalsRead",
    "compute_fundamentals",
    "fundamental_rows",
    "live_fundamentals",
    "render_fundamentals",
]
