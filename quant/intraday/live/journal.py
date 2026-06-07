"""Append-only tick journal for the intraday sleeve. Written under
data_dir/intraday/live/ (gitignored); the source of truth for status + drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

_COLS = ["ts", "sleeve_value", "day_pnl", "round_trips", "n_orders", "halted", "note"]


@dataclass(frozen=True)
class TickRecord:
    ts: datetime
    sleeve_value: float
    day_pnl: float
    round_trips: int
    n_orders: int
    halted: bool
    note: str


def _journal_path(data_dir: Path) -> Path:
    return data_dir / "intraday" / "live" / "ticks.parquet"


def append_tick(data_dir: Path, rec: TickRecord) -> None:
    path = _journal_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(rec)], columns=_COLS)
    if path.exists():
        existing = pd.read_parquet(path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_parquet(path, index=False)


def read_ticks(data_dir: Path) -> pd.DataFrame:
    path = _journal_path(data_dir)
    if not path.exists():
        return pd.DataFrame(columns=_COLS)
    return pd.read_parquet(path)
