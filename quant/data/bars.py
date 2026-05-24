"""Daily bar fetcher: Alpaca primary, yfinance backup, parquet cache.

The cache layout is per-symbol parquet files at data/raw/<symbol>.parquet.
Each file holds the full history we've seen -- append-only growth, no rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from quant.util.config import Settings
from quant.util.logging import logger

_BAR_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class BarRequest:
    """A request for daily bars over [start, end] inclusive."""

    symbols: list[str]
    start: date
    end: date
    timeframe: str = "1Day"


def _cache_path(symbol: str, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir  # type: ignore[call-arg]
    return base / "raw" / f"{symbol}.parquet"


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index, name="timestamp")
    # Normalize to seconds resolution to match what the bar-builder produces
    # (parquet may promote datetime64[s] → datetime64[ms] on round-trip).
    df.index = df.index.as_unit("s")
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _merge_cache(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _fetch_alpaca(
    symbols: list[str], start: date, end: date, settings: Settings
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars from Alpaca for the given symbols."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
    )
    bars = client.get_stock_bars(req)
    raw: pd.DataFrame = bars.df  # type: ignore[union-attr]  # alpaca BarSet -> DataFrame

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for sym in symbols:
        if sym not in raw.index.get_level_values(0):
            continue
        sym_df: pd.DataFrame = raw.xs(sym, level=0).copy()  # type: ignore[assignment]
        sym_df.index = pd.DatetimeIndex(sym_df.index.date, name="timestamp")  # type: ignore[attr-defined]
        sym_df = sym_df[[c for c in _BAR_COLUMNS if c in sym_df.columns]]
        out[sym] = sym_df
    return out


def _fetch_yfinance(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Fallback fetcher using yfinance."""
    import yfinance as yf

    raw = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    if len(symbols) == 1:
        df = raw.copy()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})
        df.index = pd.DatetimeIndex(df.index.date, name="timestamp")
        out[symbols[0]] = df[[c for c in _BAR_COLUMNS if c in df.columns]]
        return out

    for sym in symbols:
        if sym not in raw.columns.get_level_values(0):
            continue
        df = raw[sym].copy()
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.DatetimeIndex(df.index.date, name="timestamp")
        out[sym] = df[[c for c in _BAR_COLUMNS if c in df.columns]]
    return out


def get_bars(req: BarRequest) -> pd.DataFrame:
    """Return a wide DataFrame indexed by date with (symbol, field) columns.

    Cache strategy: read existing parquet, identify the gap vs the request range,
    fetch only what's missing, merge, and write back.
    """
    settings = Settings()  # type: ignore[call-arg]
    data_dir = settings.data_dir

    frames: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for symbol in req.symbols:
        path = _cache_path(symbol, data_dir)
        if not path.exists():
            to_fetch.append(symbol)
            continue
        cached = _read_cache(path)
        have_start = cached.index.min().date() if len(cached) else None
        have_end = cached.index.max().date() if len(cached) else None
        if (
            have_start is None
            or have_start > req.start
            or (have_end is not None and have_end < req.end)
        ):
            to_fetch.append(symbol)
        frames[symbol] = cached

    if to_fetch:
        try:
            fetched = _fetch_alpaca(to_fetch, req.start, req.end, settings)
        except Exception as exc:  # intentional broad catch with fallback to yfinance
            logger.warning("Alpaca fetch failed ({}); falling back to yfinance", exc)
            fetched = _fetch_yfinance(to_fetch, req.start, req.end)

        for sym, df in fetched.items():
            path = _cache_path(sym, data_dir)
            merged = _merge_cache(_read_cache(path), df) if path.exists() else df
            _write_cache(merged, path)
            frames[sym] = merged

    # Slice each frame to the requested window and stack columns
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        mask = (df.index >= pd.Timestamp(req.start)) & (df.index <= pd.Timestamp(req.end))
        sliced[sym] = df.loc[mask]

    if not sliced:
        return pd.DataFrame()
    return pd.concat(sliced, axis=1)
