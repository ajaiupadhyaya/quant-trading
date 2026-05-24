"""Live execution surface: per-strategy bookkeeping, daily rebalance, journal.

The live layer is what GitHub Actions calls every weekday at 15:55 ET. It
combines the strategy registry, the Alpaca client, and three append-only
parquet files under ``data/live/``:

  * ``equity.parquet``            — one row per rebalance: account-level snapshot
  * ``trades.parquet``            — one row per submitted order, with attribution
  * ``strategy_positions.parquet`` — long-format snapshot of target shares per (date, strategy, symbol)

Aggregate Alpaca positions are the source of truth for cash + P&L; the
per-strategy parquet is our independent bookkeeping that survives even if
Alpaca's API is briefly unavailable.
"""

from __future__ import annotations

from quant.live.bookkeeping import (
    EQUITY_COLUMNS,
    STRATEGY_POSITIONS_COLUMNS,
    TRADES_COLUMNS,
    append_equity_row,
    append_trades,
    last_strategy_positions,
    write_strategy_positions,
)
from quant.live.journal import read_journal
from quant.live.rebalance import RebalanceReport, run_rebalance

__all__ = [
    "EQUITY_COLUMNS",
    "STRATEGY_POSITIONS_COLUMNS",
    "TRADES_COLUMNS",
    "RebalanceReport",
    "append_equity_row",
    "append_trades",
    "last_strategy_positions",
    "read_journal",
    "run_rebalance",
    "write_strategy_positions",
]
