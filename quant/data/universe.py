"""S&P 500 + ETF universe lookups with on-disk snapshots."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.util.config import Settings
from quant.util.logging import logger

ETF_UNIVERSE: list[str] = ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "EFA", "EEM"]

_WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def etf_universe() -> list[str]:
    """Return the canonical 8-ETF universe used by trend-following + HRP strategies."""
    return list(ETF_UNIVERSE)


def sp500_constituents() -> list[str]:
    """Fetch the current S&P 500 ticker list from Wikipedia.

    Wikipedia uses dotted tickers (BRK.B); Alpaca + yfinance use dashed (BRK-B).
    We normalize to the dashed form so downstream calls work without symbol mapping.
    """
    logger.info("Fetching S&P 500 constituents from Wikipedia")
    tables = pd.read_html(_WIKIPEDIA_SP500_URL)
    symbols_series = tables[0]["Symbol"].astype(str)
    return [s.strip().replace(".", "-") for s in symbols_series]


def _snapshot_path(snapshot_date: date, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir  # type: ignore[call-arg]
    return base / "universe" / f"sp500_{snapshot_date.isoformat()}.csv"


def save_sp500_snapshot(
    tickers: list[str],
    snapshot_date: date,
    data_dir: Path | None = None,
) -> Path:
    """Persist a ticker list to data/universe/sp500_YYYY-MM-DD.csv."""
    path = _snapshot_path(snapshot_date, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(tickers, name="symbol").to_csv(path, index=False)
    logger.info("Saved S&P 500 snapshot to {} ({} tickers)", path, len(tickers))
    return path


def load_sp500_snapshot(
    snapshot_date: date,
    data_dir: Path | None = None,
) -> list[str]:
    """Read a previously-saved snapshot. Raises FileNotFoundError if absent."""
    path = _snapshot_path(snapshot_date, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"No S&P 500 snapshot at {path}")
    df = pd.read_csv(path)
    return df["symbol"].astype(str).tolist()
