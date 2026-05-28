"""Tests for quant.data.bars."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.data.bars import (
    BarRequest,
    _cache_path,
    _merge_cache,
    _read_cache,
    _write_cache,
    get_bars,
)


def _fake_alpaca_frame(symbol: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [1_000_000 + i for i in range(len(dates))],
        },
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="timestamp"),
    )


def test_cache_round_trip(tmp_data_dir: Path) -> None:
    df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    path = _cache_path("AAPL", tmp_data_dir)
    _write_cache(df, path)
    assert path.exists()
    loaded = _read_cache(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_get_bars_cache_hit(tmp_data_dir: Path, fake_env: None) -> None:
    df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    _write_cache(df, _cache_path("AAPL", tmp_data_dir))

    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    with patch("quant.data.bars._fetch_alpaca") as mock_alpaca:
        result = get_bars(req)
    mock_alpaca.assert_not_called()
    assert ("AAPL", "close") in result.columns
    assert len(result) == 2


def test_get_bars_cache_miss_calls_alpaca(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    with patch("quant.data.bars._fetch_alpaca", return_value={"AAPL": fake_df}) as mock_alpaca:
        result = get_bars(req)
    mock_alpaca.assert_called_once()
    assert ("AAPL", "close") in result.columns
    assert _cache_path("AAPL", tmp_data_dir).exists()


def test_get_bars_alpaca_failure_falls_back_to_yfinance(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    with (
        patch("quant.data.bars._fetch_alpaca", side_effect=RuntimeError("alpaca down")),
        patch("quant.data.bars._fetch_yfinance", return_value={"AAPL": fake_df}) as mock_yf,
    ):
        result = get_bars(req)
    mock_yf.assert_called_once()
    assert ("AAPL", "close") in result.columns


def test_merge_cache_dedups_and_sorts() -> None:
    """_merge_cache: overlapping rows resolved last-write-wins, output sorted."""
    existing = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    # New frame: day-3 overlaps (different close) + day-4 is new, presented out-of-order.
    new = pd.DataFrame(
        {
            "open": [200.0, 199.0],
            "high": [201.0, 200.0],
            "low": [199.0, 198.0],
            "close": [999.99, 200.5],
            "volume": [2_000_000, 2_000_001],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp(date(2024, 1, 4)), pd.Timestamp(date(2024, 1, 3))],
            name="timestamp",
        ),
    )
    merged = _merge_cache(existing, new)
    assert list(merged.index) == [
        pd.Timestamp(date(2024, 1, 2)),
        pd.Timestamp(date(2024, 1, 3)),
        pd.Timestamp(date(2024, 1, 4)),
    ]
    # day-3 (close=200.5) appears in `new` at position 1, so last-write-wins.
    # Existing day-3 close was 101.5 (overwritten).
    assert merged.loc[pd.Timestamp(date(2024, 1, 3)), "close"] == 200.5
    assert merged.loc[pd.Timestamp(date(2024, 1, 4)), "close"] == 999.99


def test_get_bars_partial_cache_fetches_gap_and_merges(tmp_data_dir: Path, fake_env: None) -> None:
    """Walk-forward case: cache has day-2..day-3, request extends to day-5.

    Triggers the gap-detection branch (have_end < req.end) and the merge writeback.
    """
    initial = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    _write_cache(initial, _cache_path("AAPL", tmp_data_dir))

    incremental = _fake_alpaca_frame("AAPL", [date(2024, 1, 4), date(2024, 1, 5)])
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 5))
    with patch("quant.data.bars._fetch_alpaca", return_value={"AAPL": incremental}) as mock_alpaca:
        result = get_bars(req)

    mock_alpaca.assert_called_once()
    # Result spans the full requested range: 4 rows.
    assert len(result) == 4
    # Cache on disk now contains the union of initial and incremental.
    on_disk = _read_cache(_cache_path("AAPL", tmp_data_dir))
    assert len(on_disk) == 4


def test_write_cache_drops_all_nan_rows(tmp_data_dir: Path) -> None:
    """All-NaN rows must never persist to disk — they corrupt gap detection."""
    good = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    nan_row = pd.DataFrame(
        {col: [float("nan")] for col in good.columns},
        index=pd.DatetimeIndex([pd.Timestamp(date(2024, 1, 4))], name="timestamp"),
    )
    polluted = pd.concat([good, nan_row])
    path = _cache_path("AAPL", tmp_data_dir)
    _write_cache(polluted, path)
    on_disk = _read_cache(path)
    assert len(on_disk) == 2
    assert pd.Timestamp(date(2024, 1, 4)) not in on_disk.index


def test_read_cache_drops_all_nan_rows(tmp_data_dir: Path) -> None:
    """Defensive: pre-existing parquet with NaN-only rows is cleaned on read."""
    good = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    nan_row = pd.DataFrame(
        {col: [float("nan")] for col in good.columns},
        index=pd.DatetimeIndex([pd.Timestamp(date(2024, 1, 4))], name="timestamp"),
    )
    polluted = pd.concat([good, nan_row])
    path = _cache_path("AAPL", tmp_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    polluted.to_parquet(path)  # write WITHOUT the sanitization helper
    loaded = _read_cache(path)
    assert len(loaded) == 2
    assert pd.Timestamp(date(2024, 1, 4)) not in loaded.index


def test_get_bars_refetches_when_cache_tail_is_nan(tmp_data_dir: Path, fake_env: None) -> None:
    """A cache whose tail is NaN-only must trigger a re-fetch, not look fresh."""
    good = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    nan_row = pd.DataFrame(
        {col: [float("nan")] for col in good.columns},
        index=pd.DatetimeIndex([pd.Timestamp(date(2024, 1, 5))], name="timestamp"),
    )
    polluted = pd.concat([good, nan_row])
    path = _cache_path("AAPL", tmp_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    polluted.to_parquet(path)

    incremental = _fake_alpaca_frame("AAPL", [date(2024, 1, 4), date(2024, 1, 5)])
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 5))
    with patch("quant.data.bars._fetch_alpaca", return_value={"AAPL": incremental}) as mock_alpaca:
        result = get_bars(req)

    mock_alpaca.assert_called_once()
    assert len(result) == 4
    on_disk = _read_cache(path)
    assert on_disk.notna().all().all()
    assert len(on_disk) == 4


def test_fetch_yfinance_single_symbol_handles_multiindex_columns(
    tmp_data_dir: Path,
) -> None:
    """yfinance with group_by='ticker' returns MultiIndex cols even for one symbol."""
    from quant.data.bars import _fetch_yfinance

    fake_yf_response = pd.DataFrame(
        {
            ("SPY", "Open"): [100.0, 101.0],
            ("SPY", "High"): [102.0, 103.0],
            ("SPY", "Low"): [99.0, 100.0],
            ("SPY", "Close"): [101.0, 102.0],
            ("SPY", "Adj Close"): [101.0, 102.0],
            ("SPY", "Volume"): [1_000_000, 1_100_000],
        },
        index=pd.DatetimeIndex([pd.Timestamp(date(2024, 1, 2)), pd.Timestamp(date(2024, 1, 3))]),
    )
    with patch("yfinance.download", return_value=fake_yf_response):
        out = _fetch_yfinance(["SPY"], date(2024, 1, 2), date(2024, 1, 3))

    assert "SPY" in out
    df = out["SPY"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert not df.isna().any().any()


def test_merge_cache_does_not_widen_when_new_columns_missing() -> None:
    """If `new` is missing OHLCV columns, merge must not introduce NaNs."""
    existing = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    new = pd.DataFrame(
        {},
        index=pd.DatetimeIndex([pd.Timestamp(date(2024, 1, 4))], name="timestamp"),
    )
    merged = _merge_cache(existing, new)
    assert list(merged.columns) == list(existing.columns)
    assert not merged.isna().any().any()


def test_get_bars_multi_symbol_result_shape(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL", "MSFT"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_aapl = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    fake_msft = _fake_alpaca_frame("MSFT", [date(2024, 1, 2), date(2024, 1, 3)])
    with patch(
        "quant.data.bars._fetch_alpaca",
        return_value={"AAPL": fake_aapl, "MSFT": fake_msft},
    ):
        result = get_bars(req)
    # Multi-index columns: (symbol, field)
    assert set(result.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert "close" in result.columns.get_level_values(1)
