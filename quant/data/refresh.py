"""Bulk bar-cache refresh for the union of standard + registered universes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from quant.data import universe as _universe_module
from quant.data.bars import BarRequest, get_bars
from quant.data.universe import ETF_UNIVERSE
from quant.strategies import REGISTRY
from quant.util.logging import logger


@dataclass(frozen=True)
class RefreshReport:
    symbols: list[str]
    symbols_fetched: int
    rows_total: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


def _union_universe() -> list[str]:
    """Standard universes (ETFs + S&P 500) union registered strategy universes."""
    symbols: set[str] = set(ETF_UNIVERSE)
    try:
        symbols.update(_universe_module.sp500_constituents())
    except Exception as exc:  # network-flaky fallback acceptable here
        logger.warning("Could not fetch S&P 500 constituents: {}", exc)

    for cls in REGISTRY.values():
        symbols.update(cls.spec.universe)
    return sorted(symbols)


def refresh_caches(
    start: date,
    end: date,
    *,
    chunk_size: int = 50,
) -> RefreshReport:
    """Fetch bars for the union universe over [start, end] and update the parquet cache.

    Symbols are fetched in chunks of `chunk_size` to keep individual API calls small.
    Errors are collected and returned; individual chunk failures do NOT stop the refresh.
    """
    t0 = time.monotonic()
    symbols = _union_universe()
    errors: list[str] = []
    rows_total = 0
    fetched_count = 0

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        try:
            df = get_bars(BarRequest(symbols=chunk, start=start, end=end))
            rows_total += int(df.shape[0]) if not df.empty else 0
            fetched_count += len(chunk)
            logger.info("refresh chunk {}-{}: {} symbols", i, i + len(chunk), len(chunk))
        except Exception as exc:  # chunk-level resilience
            msg = f"chunk {i}-{i + len(chunk)}: {exc!r}"
            errors.append(msg)
            logger.error(msg)

    return RefreshReport(
        symbols=symbols,
        symbols_fetched=fetched_count,
        rows_total=rows_total,
        elapsed_s=time.monotonic() - t0,
        errors=errors,
    )
