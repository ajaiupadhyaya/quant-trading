"""Tests for the combined-book backtest orchestrator."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.backtest.combined import CombinedResult, run_combined_book
from quant.backtest.engine import BacktestConfig
from tests.conftest import EqualWeightStrategy, synthetic_bars


def _two_strats() -> tuple[dict[str, EqualWeightStrategy], dict[str, pd.DataFrame]]:
    bars_a = synthetic_bars(["AAA", "BBB"], date(2022, 1, 1), date(2023, 12, 31), seed=1)
    bars_b = synthetic_bars(["AAA", "BBB"], date(2022, 1, 1), date(2023, 12, 31), seed=2)
    return (
        {"strat-a": EqualWeightStrategy(bars=bars_a), "strat-b": EqualWeightStrategy(bars=bars_b)},
        {"strat-a": bars_a, "strat-b": bars_b},
    )


def test_combined_result_shape() -> None:
    strategies, bars_per = _two_strats()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(starting_equity=200_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )
    assert isinstance(result, CombinedResult)
    assert not result.equity_curve.empty
    assert len(result.per_strategy) == 2
    assert pytest.approx(result.starting_equity) == 200_000.0


def test_default_allocation_is_equal() -> None:
    strategies, bars_per = _two_strats()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(starting_equity=200_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )
    assert result.allocation == {"strat-a": 0.5, "strat-b": 0.5}


def test_combined_equity_equals_sum_of_sub_curves() -> None:
    strategies, bars_per = _two_strats()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(starting_equity=200_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )
    sub_total = sum(r.ending_equity for r in result.per_strategy.values())
    # Tolerance: the joint curve ffills, so on shared dates the sum should match
    # the sum of per-strategy ending equities exactly.
    assert abs(result.ending_equity - sub_total) < 1e-6


def test_custom_allocation_honored() -> None:
    strategies, bars_per = _two_strats()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(starting_equity=100_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
        allocation={"strat-a": 0.75, "strat-b": 0.25},
    )
    assert result.allocation == {"strat-a": 0.75, "strat-b": 0.25}
    sub_a = result.per_strategy["strat-a"]
    sub_b = result.per_strategy["strat-b"]
    # Sub-strategies should have been given proportional starting equity.
    assert sub_a.starting_equity == pytest.approx(75_000.0)
    assert sub_b.starting_equity == pytest.approx(25_000.0)


def test_trades_carry_strategy_slug_attribution() -> None:
    strategies, bars_per = _two_strats()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(starting_equity=100_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )
    if not result.trades.empty:
        assert "strategy_slug" in result.trades.columns
        assert set(result.trades["strategy_slug"].unique()).issubset({"strat-a", "strat-b"})


def test_empty_strategies_returns_empty_result() -> None:
    result = run_combined_book(
        strategies={},
        bars_per_strategy={},
        config=BacktestConfig(starting_equity=100_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )
    assert result.equity_curve.empty
    assert result.ending_equity == 100_000.0


def test_mismatched_keys_raises() -> None:
    strategies, bars_per = _two_strats()
    bars_per.pop("strat-b")  # introduce key mismatch
    with pytest.raises(ValueError, match="identical keys"):
        run_combined_book(
            strategies=strategies,
            bars_per_strategy=bars_per,
            config=BacktestConfig(),
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
        )


def test_allocation_must_sum_to_one() -> None:
    strategies, bars_per = _two_strats()
    with pytest.raises(ValueError, match=r"sum to ~1\.0"):
        run_combined_book(
            strategies=strategies,
            bars_per_strategy=bars_per,
            config=BacktestConfig(),
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
            allocation={"strat-a": 0.3, "strat-b": 0.3},
        )


def test_write_combined_tearsheet_renders_html(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from quant.backtest import write_combined_tearsheet

    strategies, bars_per = _two_strats()
    result = run_combined_book(
        strategies=strategies,
        bars_per_strategy=bars_per,
        config=BacktestConfig(starting_equity=200_000.0),
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )
    out_dir = tmp_path / "combined"
    html_path = write_combined_tearsheet(result=result, out_dir=out_dir)
    assert html_path.exists()
    assert (out_dir / "equity.parquet").exists()
    html = html_path.read_text()
    assert "Combined Book" in html
    assert "Per-strategy breakdown" in html
    # Each sub-strategy slug should appear in the breakdown table.
    for slug in result.per_strategy:
        assert slug in html


def test_allocation_missing_keys_raises() -> None:
    strategies, bars_per = _two_strats()
    with pytest.raises(ValueError, match="missing keys"):
        run_combined_book(
            strategies=strategies,
            bars_per_strategy=bars_per,
            config=BacktestConfig(),
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
            allocation={"strat-a": 1.0},
        )
