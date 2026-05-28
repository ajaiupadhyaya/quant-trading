"""Tests for immutable data snapshots and data quality gates."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.data.quality import evaluate_bar_quality
from quant.data.snapshot import create_data_snapshot


def _write_raw(data_dir: Path, symbol: str, df: pd.DataFrame) -> None:
    path = data_dir / "raw" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_create_data_snapshot_hashes_raw_inputs(tmp_data_dir: Path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=3)
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [2, 3, 4], "low": [1, 2, 3], "close": [2, 3, 4], "volume": [10, 10, 10]},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )
    _write_raw(tmp_data_dir, "SPY", df)

    manifest = create_data_snapshot(
        tmp_data_dir,
        symbols=["SPY"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 6),
        snapshot_id="unit-snap",
    )

    assert manifest.snapshot_id == "unit-snap"
    assert manifest.symbols["SPY"].sha256
    assert (tmp_data_dir / "snapshots" / "unit-snap" / "manifest.json").exists()


def test_data_quality_flags_missing_duplicate_and_impossible_ohlc() -> None:
    idx = pd.DatetimeIndex(
        ["2026-01-02", "2026-01-02", "2026-01-06"],
        name="timestamp",
    )
    bars = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 9.0],
            "low": [9.0, 9.0, 10.0],
            "close": [10.5, 10.5, 10.0],
            "volume": [100, 100, 100],
        },
        index=idx,
    )

    report = evaluate_bar_quality({"SPY": bars}, start=date(2026, 1, 2), end=date(2026, 1, 7))

    assert report.symbols["SPY"].duplicate_timestamps == 1
    assert report.symbols["SPY"].missing_bars >= 1
    assert report.symbols["SPY"].impossible_ohlc == 1
    assert not report.passed


def test_data_quality_does_not_count_market_holidays_as_missing() -> None:
    idx = pd.DatetimeIndex(["2026-01-16", "2026-01-20"], name="timestamp")
    bars = pd.DataFrame(
        {
            "open": [10.0, 10.5],
            "high": [11.0, 11.0],
            "low": [9.0, 10.0],
            "close": [10.5, 10.75],
            "volume": [100, 100],
        },
        index=idx,
    )

    report = evaluate_bar_quality({"SPY": bars}, start=date(2026, 1, 16), end=date(2026, 1, 20))

    assert report.symbols["SPY"].missing_bars == 0
    assert report.passed
