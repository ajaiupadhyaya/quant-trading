"""Fail-open contract: degraded inputs degrade the record, they never raise."""

from __future__ import annotations

import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.research import signals as sig
from quant.research.signals import (
    MarketSignals,
    _avg_pairwise_corr_series,
    build_market_signals,
    load_market_signals,
    to_json_dict,
)
from tests.research.conftest import close_panel, macro_series


def _macros(index: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        macro_series(18.0, index),
        macro_series(4.0, index),
        macro_series(4.2, index),
    )


def test_empty_frame_degrades_not_raises() -> None:
    rec = build_market_signals(
        closes=pd.DataFrame(), vix=None, dgs10=None, dgs2=None, asof=date(2024, 6, 3)
    )
    assert isinstance(rec, MarketSignals)
    assert rec.computable is False
    assert rec.degraded == ("no_bars",)


def test_one_row_fail_opens() -> None:
    panel = close_panel().iloc[:1]
    rec = build_market_signals(
        closes=panel, vix=None, dgs10=None, dgs2=None, asof=panel.index[-1].date()
    )
    assert isinstance(rec, MarketSignals)
    # No history -> every windowed signal is None and the composite is uncomputable.
    assert rec.composite_score is None


def test_all_nan_close_fail_opens() -> None:
    panel = close_panel()
    panel.iloc[:, :] = np.nan
    rec = build_market_signals(
        closes=panel, vix=None, dgs10=None, dgs2=None, asof=panel.index[-1].date()
    )
    assert rec.computable is False  # all columns dropped as all-NaN


def test_missing_vix_degrades_component_not_battery() -> None:
    panel = close_panel(seed=2)
    _, d10, d2 = _macros(panel.index)
    rec = build_market_signals(
        closes=panel, vix=None, dgs10=d10, dgs2=d2, asof=panel.index[-1].date()
    )
    assert rec.vol is not None
    assert rec.vol.spy_realized_vol_ann is not None  # vol still computes
    assert rec.vol.vix_level is None  # only the VIX-derived fields drop
    assert rec.computable is True


def test_serialized_warmup_record_has_no_nan_or_inf() -> None:
    panel = close_panel().iloc[:80]  # short -> many None fields, some warmup
    rec = build_market_signals(
        closes=panel, vix=None, dgs10=None, dgs2=None, asof=panel.index[-1].date()
    )
    # allow_nan=False raises if any NaN/Inf slipped past the _finite gate.
    json.dumps(to_json_dict(rec), allow_nan=False)


def test_corr_lt2_cols_returns_none_without_warning() -> None:
    one_col = close_panel(symbols=["SPY"]).pct_change()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        s = _avg_pairwise_corr_series(one_col, 63)
    assert s.isna().all()


def test_loader_never_fetches_network(
    tmp_data_dir: object, fake_env: object, make_bars: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a warm cache, the loader must read parquet only — never call a fetcher."""
    from quant.data import bars
    from quant.data.universe import ETF_UNIVERSE

    panel_bars = make_bars(list(ETF_UNIVERSE), date(2022, 1, 3), date(2024, 6, 28), seed=1)  # type: ignore[operator]
    for sym in ETF_UNIVERSE:
        bars._write_cache(panel_bars[sym], bars._cache_path(sym))

    def _boom(*a: object, **k: object) -> object:
        raise AssertionError("network fetch attempted")

    monkeypatch.setattr(bars, "_fetch_alpaca", _boom)
    monkeypatch.setattr(bars, "_fetch_yfinance", _boom)
    monkeypatch.setattr(
        sig.macro,
        "get_series",
        lambda code: macro_series(18.0, pd.bdate_range("2022-01-03", periods=600)),
    )

    rec = load_market_signals(asof=date(2024, 6, 27))
    assert rec.computable is True


def test_loader_does_not_request_future_bars(
    tmp_data_dir: object, fake_env: object, make_bars: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant.data import bars
    from quant.data.universe import ETF_UNIVERSE

    panel_bars = make_bars(list(ETF_UNIVERSE), date(2022, 1, 3), date(2024, 6, 28), seed=1)  # type: ignore[operator]
    for sym in ETF_UNIVERSE:
        bars._write_cache(panel_bars[sym], bars._cache_path(sym))

    captured: dict[str, object] = {}
    real = bars.get_bars

    def _spy(req: object) -> object:
        captured["req"] = req
        return real(req)

    monkeypatch.setattr(sig.bars, "get_bars", _spy)
    monkeypatch.setattr(sig.macro, "get_series", lambda code: None)

    asof = date(2024, 3, 15)
    rec = load_market_signals(asof=asof)
    req = captured["req"]
    assert req.end <= asof  # type: ignore[attr-defined]
    for a in rec.assets:
        assert a is not None


def test_loader_timeout_fail_opens(
    tmp_data_dir: object, fake_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import concurrent.futures

    def _timeout(*a: object, **k: object) -> object:
        raise concurrent.futures.TimeoutError("budget exceeded")

    monkeypatch.setattr(sig.bars, "get_bars", _timeout)
    rec = load_market_signals(asof=date(2024, 6, 3))
    assert rec.computable is False
    assert rec.degraded == ("bars_error",)


@pytest.mark.parametrize("mode", ["raise", "empty"])
def test_loader_never_raises_property(
    tmp_data_dir: object, fake_env: object, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    def _bars(req: object) -> pd.DataFrame:
        if mode == "raise":
            raise RuntimeError("boom")
        return pd.DataFrame()

    monkeypatch.setattr(sig.bars, "get_bars", _bars)
    monkeypatch.setattr(sig.macro, "get_series", lambda code: (_ for _ in ()).throw(RuntimeError()))
    rec = load_market_signals(asof=date(2024, 6, 3))
    assert isinstance(rec, MarketSignals)
    assert rec.computable is False
