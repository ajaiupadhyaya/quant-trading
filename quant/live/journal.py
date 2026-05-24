"""Read the trade journal for the CLI ``quant journal`` subcommand."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.live.bookkeeping import read_trades


def read_journal(
    data_dir: Path,
    since: date | None = None,
    strategy: str | None = None,
) -> pd.DataFrame:
    """Return the trade log filtered by date / strategy."""
    df = read_trades(data_dir)
    if df.empty:
        return df
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    if strategy is not None:
        df = df[df["strategy"] == strategy]
    return df.reset_index(drop=True)
