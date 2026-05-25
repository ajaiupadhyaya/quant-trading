"""Tests for the OOS holdout + cost-sensitivity additions to run_validation."""

from __future__ import annotations

from datetime import date

from quant.backtest.cpcv import CPCVConfig
from quant.backtest.engine import BacktestConfig
from quant.backtest.validation import run_validation
from quant.backtest.walkforward import run_walkforward
from tests.conftest import EqualWeightStrategy, synthetic_bars


def _factory(params, bars):  # type: ignore[no-untyped-def]
    return EqualWeightStrategy(bars=bars, params=params)


def test_holdout_result_populated_when_window_provided() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2024, 12, 31), seed=11)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2023, 6, 30),
        config=BacktestConfig(),
        train_years=2,
        test_years=1,
        step_months=12,
    )
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=_factory,
        chosen_params={},
        backtest_config=BacktestConfig(),
        cpcv_config=CPCVConfig(n_groups=4, k_test=2),
        bootstrap_resamples=50,
        holdout_start=date(2023, 7, 1),
        holdout_end=date(2024, 12, 31),
        cost_sensitivity_bps=(0.0, 5.0),
    )
    assert report.holdout is not None
    assert report.holdout.start == date(2023, 7, 1)
    assert report.holdout.end == date(2024, 12, 31)
    assert report.holdout.n_days > 0
    # gate_holdout is just total_return > 0 — assert it has a definite bool.
    assert isinstance(report.gate_holdout, bool)


def test_no_holdout_means_gate_passes_vacuously() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2024, 12, 31), seed=12)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2024, 12, 31),
        config=BacktestConfig(),
        train_years=2,
        test_years=1,
        step_months=12,
    )
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=_factory,
        chosen_params={},
        backtest_config=BacktestConfig(),
        cpcv_config=CPCVConfig(n_groups=4, k_test=2),
        bootstrap_resamples=50,
        holdout_start=None,
        holdout_end=None,
        cost_sensitivity_bps=(),
    )
    assert report.holdout is None
    assert report.gate_holdout is True


def test_cost_sensitivity_returns_one_row_per_bps() -> None:
    bars = synthetic_bars(["AAA", "BBB"], date(2018, 1, 1), date(2023, 12, 31), seed=13)
    wf = run_walkforward(
        strategy_factory=_factory,
        param_grid={},
        bars=bars,
        start=date(2018, 1, 1),
        end=date(2023, 12, 31),
        config=BacktestConfig(),
        train_years=2,
        test_years=1,
        step_months=12,
    )
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=_factory,
        chosen_params={},
        backtest_config=BacktestConfig(),
        cpcv_config=CPCVConfig(n_groups=4, k_test=2),
        bootstrap_resamples=50,
        cost_sensitivity_bps=(0.0, 5.0, 30.0),
    )
    assert len(report.cost_sensitivity) == 3
    assert [r.slippage_bps for r in report.cost_sensitivity] == [0.0, 5.0, 30.0]
    # Higher slippage should produce lower or equal total return.
    sorted_by_bps = sorted(report.cost_sensitivity, key=lambda r: r.slippage_bps)
    rets = [r.total_return for r in sorted_by_bps]
    assert rets[0] >= rets[-1] - 1e-6
