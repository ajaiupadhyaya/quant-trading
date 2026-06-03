"""FRED macro series fetcher with parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fredapi import Fred

from quant.util.config import Settings
from quant.util.logging import logger

FRED_SERIES: dict[str, str] = {
    "vix": "VIXCLS",
    "vix3m": "VXVCLS",  # 3-month VIX (term-structure numerator)
    "tenyear": "DGS10",
    "twoyear": "DGS2",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "fedfunds": "DFF",
    "gdp": "GDPC1",
    # policy / macro-risk environment (Phase 7C)
    "epu": "USEPUINDXD",  # US Economic Policy Uncertainty (daily)
    "nfci": "NFCI",  # Chicago Fed financial conditions (neg=loose, pos=tight)
    "finstress": "STLFSI4",  # St Louis Fed financial stress index
}


def _cache_path(series_id: str, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir  # type: ignore[call-arg]
    return base / "macro" / f"{series_id}.parquet"


def get_series(series_id: str) -> pd.Series:
    """Fetch a FRED series. Returns from cache if present, else fetches and caches."""
    settings = Settings()  # type: ignore[call-arg]
    path = _cache_path(series_id, settings.data_dir)
    if path.exists():
        logger.debug("Macro cache hit: {}", series_id)
        return pd.read_parquet(path)[series_id]

    logger.info("Fetching FRED series {}", series_id)
    fred = Fred(api_key=settings.fred_api_key)
    series = fred.get_series(series_id)
    series.name = series_id

    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name=series_id).to_parquet(path)
    return series  # type: ignore[no-any-return]


def vix() -> pd.Series:
    """CBOE VIX (close)."""
    return get_series(FRED_SERIES["vix"])


def tenyear_yield() -> pd.Series:
    """10-year Treasury constant maturity rate."""
    return get_series(FRED_SERIES["tenyear"])


def unemployment_rate() -> pd.Series:
    """Civilian unemployment rate (UNRATE)."""
    return get_series(FRED_SERIES["unemployment"])


def cpi() -> pd.Series:
    """Consumer price index (CPIAUCSL, seasonally adjusted)."""
    return get_series(FRED_SERIES["cpi"])
