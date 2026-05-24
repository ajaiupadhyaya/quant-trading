"""Tests for quant.data.macro."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from quant.data.macro import (
    FRED_SERIES,
    _cache_path,
    cpi,
    get_series,
    tenyear_yield,
    unemployment_rate,
    vix,
)


def _fake_series() -> pd.Series:
    return pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
        name="VIXCLS",
    )


def test_get_series_caches_to_parquet(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred):
        s = get_series("VIXCLS")
    assert _cache_path("VIXCLS", tmp_data_dir).exists()
    assert len(s) == 3


def test_get_series_uses_cache_on_second_call(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred) as fred_cls:
        get_series("VIXCLS")
        get_series("VIXCLS")  # should hit cache
    fred_cls.assert_called_once()


def test_vix_uses_vixcls_series_id(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred):
        vix()
    fred.get_series.assert_called_with(FRED_SERIES["vix"])


def test_helpers_dispatch_correct_series(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred):
        tenyear_yield()
        unemployment_rate()
        cpi()
    called_ids = [call.args[0] for call in fred.get_series.call_args_list]
    assert FRED_SERIES["tenyear"] in called_ids
    assert FRED_SERIES["unemployment"] in called_ids
    assert FRED_SERIES["cpi"] in called_ids
