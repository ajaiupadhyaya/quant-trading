"""Daily sleeve recon: the LIVE side of the drift picture. Summarizes journaled
ticks and reconciles ledger vs broker sleeve positions. The backtest-vs-live drift
comparison needs the intraday backtest baseline (offline pipeline) and is deferred."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant.intraday.live.config import SleeveConfig
from quant.intraday.live.journal import read_ticks


def summarize_day(data_dir: Path) -> dict[str, Any]:
    df = read_ticks(data_dir)
    if df.empty:
        return {"n_ticks": 0, "last_day_pnl": 0.0, "max_round_trips": 0, "halted_any": False}
    return {
        "n_ticks": len(df),
        "last_day_pnl": float(df.iloc[-1]["day_pnl"]),
        "max_round_trips": int(df["round_trips"].max()),
        "halted_any": bool(df["halted"].any()),
    }


def position_mismatches(
    ledger_positions: dict[str, int],
    broker: Any,
    config: SleeveConfig,
) -> dict[str, tuple[int, int]]:
    """Return {symbol: (ledger_qty, broker_qty)} for sleeve-universe symbols whose
    ledger and broker positions disagree. Symbols outside the sleeve universe (i.e.
    the daily system's holdings) are ignored."""
    broker_qty = {p.symbol: int(p.qty) for p in broker.positions() if p.symbol in config.universe}
    out: dict[str, tuple[int, int]] = {}
    for sym in config.universe:
        lq = int(ledger_positions.get(sym, 0))
        bq = broker_qty.get(sym, 0)
        if lq != bq:
            out[sym] = (lq, bq)
    return out
