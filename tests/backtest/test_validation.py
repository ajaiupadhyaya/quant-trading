from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from quant.backtest.cpcv import CPCVConfig
from quant.backtest.engine import BacktestConfig
from quant.backtest.validation import (
    THRESHOLDS,
    ValidationReport,
    run_validation,
)
from quant.backtest.walkforward import run_walkforward
from quant.strategies.base import Strategy, StrategySpec
from tests.conftest import synthetic_bars


class _EqualWeightFixture(Strategy):
    spec = StrategySpec(
        slug="validation-test-eqw",
        name="EqualWeightFixture",
        description="Test fixture",
        universe=["AAA", "BBB", "CCC"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict] = {"slot": 0}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({s: 1.0 for s in self.spec.universe})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 10, "BBB": 10, "CCC": 10}


@pytest.fixture(scope="module")
def wf_result_and_bars():
    bars = synthetic_bars(
        ["AAA", "BBB", "CCC"], date(2010, 1, 1), date(2020, 12, 31), drift=0.0005
    )

    def factory(params: dict, bars: pd.DataFrame) -> Strategy:
        return _EqualWeightFixture(params=params)

    bt_cfg = BacktestConfig(starting_equity=100_000.0)
    wf = run_walkforward(
        strategy_factory=factory,
        param_grid={"slot": [0, 1, 2]},
        bars=bars,
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        config=bt_cfg,
        train_years=5,
        test_years=1,
        step_months=12,
    )
    return wf, bars, factory


def test_run_validation_returns_report_with_all_fields(wf_result_and_bars) -> None:
    wf, bars, factory = wf_result_and_bars
    report = run_validation(
        wf_result=wf,
        bars=bars,
        strategy_factory=factory,
        chosen_params={"slot": 0},
        backtest_config=BacktestConfig(starting_equity=100_000.0),
        cpcv_config=CPCVConfig(n_groups=4, k_test=2, embargo_days=0),
        bootstrap_resamples=100,
        seed=0,
    )
    assert isinstance(report, ValidationReport)
    assert report.deflated_sharpe >= 0.0
    assert report.probabilistic_sharpe >= 0.0
    assert report.bootstrap_ci is not None
    assert len(report.regime_breakdown) == 5  # five regimes
    assert isinstance(report.passed, bool)


def test_thresholds_match_spec() -> None:
    assert THRESHOLDS.deflated_sharpe == 0.3
    assert THRESHOLDS.probabilistic_sharpe == 0.7
    assert THRESHOLDS.min_positive_regimes == 3


def test_report_passed_is_and_of_four_gates() -> None:
    report = ValidationReport(
        deflated_sharpe=0.5,
        probabilistic_sharpe=0.8,
        bootstrap_ci=None,  # treated as fail when missing
        regime_breakdown=[],
        cpcv_path_sharpes=np.array([]),
        n_positive_regimes=4,
        trial_sharpes=np.array([0.1]),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=False,
        gate_regime=True,
    )
    assert report.passed is False


def test_report_passed_true_when_all_gates_true() -> None:
    report = ValidationReport(
        deflated_sharpe=0.5,
        probabilistic_sharpe=0.8,
        bootstrap_ci=None,
        regime_breakdown=[],
        cpcv_path_sharpes=np.array([]),
        n_positive_regimes=4,
        trial_sharpes=np.array([0.1]),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
    )
    assert report.passed is True
