# quant/intraday/data/config.py
"""Intraday data-layer configuration: universe, storage roots, partition paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# ~100 most-liquid US names (large-caps + major ETFs). Curated for tight spreads;
# intraday edge requires liquidity. Point-in-time membership lives in universe.py.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    # major ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "EEM", "EFA", "XLF", "XLK", "XLE",
    "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "GLD", "SLV", "TLT",
    "HYG", "LQD", "VXX", "SQQQ", "TQQQ", "ARKK", "SMH", "SOXL",
    # mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "ADBE", "CRM", "ORCL", "INTC", "CSCO", "QCOM", "TXN", "MU", "PLTR",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK", "V", "MA",
    # healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "BMY",
    # consumer / industrial / energy
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "DIS",
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "XOM", "CVX", "COP", "SLB",
    # comm / other liquid
    "T", "VZ", "CMCSA", "F", "GM", "UBER", "ABNB", "COIN", "SHOP", "SNOW",
    "DKNG", "RIVN", "SOFI", "MARA",
)


def partition_path(root: Path, dataset: str, symbol: str, day: date) -> Path:
    """root/<dataset>/symbol=<SYM>/date=<YYYY-MM-DD>.parquet (Hive-style)."""
    return root / dataset / f"symbol={symbol}" / f"date={day.isoformat()}.parquet"


@dataclass(frozen=True)
class IntradayConfig:
    data_root: Path
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    hot_window_days: int = 5  # rolling lookback the live engine keeps in memory
    _unused: tuple[()] = field(default=(), repr=False)
