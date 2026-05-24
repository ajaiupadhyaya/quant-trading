"""Validation orchestrator: runs the full §4 battery and emits a pass/fail report.

Consumes a WalkforwardResult plus the original bars+factory and produces
booleans for the four pass-live criteria.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant.backtest.bootstrap import BootstrapCI, bootstrap_ci
from quant.backtest.cpcv import CPCVConfig, run_cpcv
from quant.backtest.dsr import deflated_sharpe, probabilistic_sharpe
from quant.backtest.engine import BacktestConfig
from quant.backtest.regimes import (
    RegimeBreakdown,
    compute_regime_breakdown,
    count_positive_regimes,
)
from quant.backtest.walkforward import WalkforwardResult

if TYPE_CHECKING:
    from quant.strategies.base import Strategy


StrategyFactory = Callable[[dict[str, Any], pd.DataFrame], "Strategy"]


@dataclass(frozen=True)
class _Thresholds:
    deflated_sharpe: float = 0.3
    probabilistic_sharpe: float = 0.7
    min_positive_regimes: int = 3


THRESHOLDS = _Thresholds()


@dataclass(frozen=True)
class ValidationReport:
    deflated_sharpe: float
    probabilistic_sharpe: float
    bootstrap_ci: BootstrapCI | None
    regime_breakdown: list[RegimeBreakdown]
    cpcv_path_sharpes: np.ndarray
    n_positive_regimes: int
    trial_sharpes: np.ndarray
    gate_deflated_sharpe: bool
    gate_probabilistic_sharpe: bool
    gate_bootstrap_lower: bool
    gate_regime: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.gate_deflated_sharpe
            and self.gate_probabilistic_sharpe
            and self.gate_bootstrap_lower
            and self.gate_regime
        )


def _trial_sharpes_from_cpcv(cpcv_paths: np.ndarray) -> np.ndarray:
    """Convert CPCV per-path Sharpes from annualized to per-period scale.

    walk-forward Sharpes are annualized (multiplied by sqrt(252)). DSR's
    formula expects per-period Sharpes, so we divide back out.
    """
    if len(cpcv_paths) == 0:
        return cpcv_paths
    return np.asarray(cpcv_paths / np.sqrt(252.0))


def run_validation(
    wf_result: WalkforwardResult,
    bars: pd.DataFrame,
    strategy_factory: StrategyFactory,
    chosen_params: dict[str, Any],
    backtest_config: BacktestConfig,
    cpcv_config: CPCVConfig | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_block_len: int = 5,
    seed: int = 0,
) -> ValidationReport:
    """Run DSR, PSR, bootstrap, regimes, and CPCV; build the pass-fail report."""
    if cpcv_config is None:
        cpcv_config = CPCVConfig()
    oos_returns = wf_result.oos_returns

    # CPCV path Sharpes feed DSR's trial count + variance.
    if len(oos_returns) > 0:
        cpcv = run_cpcv(
            strategy_factory=strategy_factory,
            params=chosen_params,
            bars=bars,
            start=oos_returns.index.min().date(),
            end=oos_returns.index.max().date(),
            backtest_config=backtest_config,
            cpcv_config=cpcv_config,
        )
        cpcv_paths = cpcv.path_sharpes
    else:
        cpcv_paths = np.array([], dtype=float)

    trial_sharpes = _trial_sharpes_from_cpcv(cpcv_paths)

    psr = probabilistic_sharpe(oos_returns, sr_benchmark=0.0)
    dsr = deflated_sharpe(oos_returns, trial_sharpes=trial_sharpes)

    if len(oos_returns) > 0:
        ci: BootstrapCI | None = bootstrap_ci(
            oos_returns,
            n_resamples=bootstrap_resamples,
            mean_block_len=bootstrap_block_len,
            seed=seed,
        )
    else:
        ci = None

    breakdown = compute_regime_breakdown(oos_returns)
    n_positive = count_positive_regimes(breakdown)

    gate_dsr = dsr >= THRESHOLDS.deflated_sharpe
    gate_psr = psr >= THRESHOLDS.probabilistic_sharpe
    gate_boot = ci is not None and ci.total_return_p05 > 0.0
    gate_regime = n_positive >= THRESHOLDS.min_positive_regimes

    return ValidationReport(
        deflated_sharpe=dsr,
        probabilistic_sharpe=psr,
        bootstrap_ci=ci,
        regime_breakdown=breakdown,
        cpcv_path_sharpes=cpcv_paths,
        n_positive_regimes=n_positive,
        trial_sharpes=trial_sharpes,
        gate_deflated_sharpe=gate_dsr,
        gate_probabilistic_sharpe=gate_psr,
        gate_bootstrap_lower=gate_boot,
        gate_regime=gate_regime,
    )
