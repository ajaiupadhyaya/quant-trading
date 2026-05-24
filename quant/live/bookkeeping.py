"""Append-only parquet writers/readers for ``data/live/``.

Three files, all in long format so they can grow forever without ever
rewriting earlier rows:

* ``equity.parquet`` — account snapshot per rebalance
* ``trades.parquet`` — one row per order we submitted
* ``strategy_positions.parquet`` — per-strategy target shares snapshot

All readers gracefully return empty frames with the expected columns when the
underlying parquet doesn't exist yet. All writers create parent directories on
demand.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

EQUITY_COLUMNS: list[str] = [
    "date",
    "equity",
    "last_equity",
    "cash",
    "buying_power",
    "portfolio_value",
]

TRADES_COLUMNS: list[str] = [
    "date",
    "strategy",
    "symbol",
    "side",
    "qty",
    "client_order_id",
    "dry_run",
]

STRATEGY_POSITIONS_COLUMNS: list[str] = [
    "date",
    "strategy",
    "symbol",
    "qty",
]


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


def _live_dir(data_dir: Path) -> Path:
    p = data_dir / "live"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_parquet(path: Path, new: pd.DataFrame, cols: list[str]) -> None:
    if new.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new[cols]], ignore_index=True)
    else:
        combined = new[cols].reset_index(drop=True)
    combined.to_parquet(path, index=False)


def append_equity_row(
    data_dir: Path,
    *,
    asof: date,
    equity: float,
    last_equity: float,
    cash: float,
    buying_power: float,
    portfolio_value: float,
) -> Path:
    """Append a single equity snapshot row to ``live/equity.parquet``."""
    path = _live_dir(data_dir) / "equity.parquet"
    row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(asof),
                "equity": float(equity),
                "last_equity": float(last_equity),
                "cash": float(cash),
                "buying_power": float(buying_power),
                "portfolio_value": float(portfolio_value),
            }
        ]
    )
    _append_parquet(path, row, EQUITY_COLUMNS)
    return path


def append_trades(
    data_dir: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Append one row per submitted order to ``live/trades.parquet``.

    Each row dict must include the keys in TRADES_COLUMNS. Empty list is a no-op.
    """
    path = _live_dir(data_dir) / "trades.parquet"
    if not rows:
        return path
    frame = pd.DataFrame(rows)
    missing = set(TRADES_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"trades rows missing columns: {sorted(missing)}")
    _append_parquet(path, frame, TRADES_COLUMNS)
    return path


def read_equity(data_dir: Path) -> pd.DataFrame:
    """Return all equity snapshots, sorted oldest -> newest."""
    path = _live_dir(data_dir) / "equity.parquet"
    if not path.exists():
        return _empty(EQUITY_COLUMNS)
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def read_trades(data_dir: Path) -> pd.DataFrame:
    """Return all submitted-order rows, sorted oldest -> newest."""
    path = _live_dir(data_dir) / "trades.parquet"
    if not path.exists():
        return _empty(TRADES_COLUMNS)
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def write_strategy_positions(
    data_dir: Path,
    asof: date,
    strategy_slug: str,
    target: dict[str, int],
) -> Path:
    """Append a per-strategy positions snapshot for (asof, strategy_slug)."""
    path = _live_dir(data_dir) / "strategy_positions.parquet"
    rows = [
        {
            "date": pd.Timestamp(asof),
            "strategy": strategy_slug,
            "symbol": sym,
            "qty": int(qty),
        }
        for sym, qty in target.items()
    ]
    if not rows:
        return path
    _append_parquet(path, pd.DataFrame(rows), STRATEGY_POSITIONS_COLUMNS)
    return path


def last_strategy_positions(
    data_dir: Path,
    strategy_slug: str,
) -> dict[str, int]:
    """Return the most-recent snapshot for ``strategy_slug`` (or {} if none)."""
    path = _live_dir(data_dir) / "strategy_positions.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty or "strategy" not in df.columns:
        return {}
    df = df[df["strategy"] == strategy_slug]
    if df.empty:
        return {}
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date]
    return {str(r.symbol): int(r.qty) for r in snap.itertuples(index=False)}
