"""Tests for write_tearsheet."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant.backtest.engine import BacktestConfig
from quant.backtest.tearsheet import write_tearsheet
from quant.backtest.walkforward import run_walkforward
from quant.strategies.base import Strategy
from tests.conftest import EqualWeightStrategy


def _factory(params: dict[str, Any], bars: pd.DataFrame) -> Strategy:
    return EqualWeightStrategy(bars=bars, params=params)


def test_tearsheet_writes_three_files(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    assert (out_dir / "tearsheet.html").exists()
    assert (out_dir / "walkforward.parquet").exists()
    assert (out_dir / "chosen_params.json").exists()


def test_tearsheet_html_contains_strategy_name(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    html = (out_dir / "tearsheet.html").read_text()
    assert "Equal Weight (test)" in html
    # Embedded charts use base64 data URIs:
    assert "data:image/png;base64," in html
    # Key metrics:
    assert "Sharpe" in html
    assert "Max Drawdown" in html
    assert "CAGR" in html


def test_tearsheet_walkforward_parquet_matches_oos_curve(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    on_disk = pd.read_parquet(out_dir / "walkforward.parquet")
    assert "equity" in on_disk.columns
    assert len(on_disk) == len(wf.oos_equity_curve)


def test_tearsheet_chosen_params_json_shape(
    tmp_path: Path,
    make_bars: Callable[..., pd.DataFrame],
) -> None:
    bars = make_bars(["AAA", "BBB"], date(2010, 1, 1), date(2020, 12, 31), seed=0)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={"_dummy": [1]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=BacktestConfig(),
    )

    out_dir = tmp_path / "backtests" / "equal-weight-test"
    write_tearsheet(
        result=wf,
        slug="equal-weight-test",
        strategy_name="Equal Weight (test)",
        out_dir=out_dir,
    )

    payload = json.loads((out_dir / "chosen_params.json").read_text())
    assert "windows" in payload
    assert isinstance(payload["windows"], list)
    assert len(payload["windows"]) > 0
    first = payload["windows"][0]
    for key in ("train_start", "train_end", "test_start", "test_end", "params"):
        assert key in first


def test_tearsheet_empty_walkforward(tmp_path: Path) -> None:
    """A walk-forward with no fit-able windows produces a minimal tear-sheet, no crash."""
    from quant.backtest.engine import BacktestResult
    from quant.backtest.walkforward import WalkforwardResult

    empty_series = pd.Series(dtype=float, name="equity")
    empty_trades = pd.DataFrame(
        columns=[
            "date",
            "symbol",
            "side",
            "qty",
            "fill_price",
            "slippage_cost",
            "commission_cost",
            "strategy_slug",
        ]
    )
    empty_combined = BacktestResult(
        equity_curve=empty_series,
        returns=empty_series,
        positions=pd.DataFrame(),
        trades=empty_trades,
        config=BacktestConfig(),
        starting_equity=100_000.0,
        ending_equity=100_000.0,
    )
    empty_wf = WalkforwardResult(
        oos_equity_curve=empty_series,
        oos_returns=empty_series,
        oos_trades=empty_trades,
        per_window_params=[],
        combined_result=empty_combined,
    )
    out_dir = tmp_path / "backtests" / "empty"
    write_tearsheet(result=empty_wf, slug="empty", strategy_name="Empty", out_dir=out_dir)
    assert (out_dir / "tearsheet.html").exists()
    html = (out_dir / "tearsheet.html").read_text()
    assert "no walk-forward windows" in html.lower()
