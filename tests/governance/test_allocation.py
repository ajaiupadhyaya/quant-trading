"""Tests for governance capital allocation."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant.governance.allocation import (
    AllocationConfig,
    allocate_capital,
    hrp_raw_weights,
    load_strategy_returns,
    risk_based_raw_weights,
    strategy_risk,
)
from quant.governance.models import GovernanceState, StrategyState, ValidationEvidence


def _state(slug: str, state: GovernanceState) -> StrategyState:
    return StrategyState(
        slug=slug,
        state=state,
        evaluated_at=datetime(2026, 5, 27),
        validation_age_days=0,
        reason_codes=[] if state is GovernanceState.LIVE else ["failed_gate_bootstrap_lower"],
        reason="",
        code_enabled_live=True,
    )


def _evidence(slug: str, dsr: float) -> ValidationEvidence:
    from datetime import date

    return ValidationEvidence(
        slug=slug,
        run_date=date(2026, 5, 27),
        data_start=date(2010, 1, 1),
        data_end=date(2026, 5, 26),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
        gate_holdout=True,
        deflated_sharpe=dsr,
        probabilistic_sharpe=0.9,
        bootstrap_total_return_p05=0.02,
        n_positive_regimes=4,
        n_tested_regimes=4,
        holdout_total_return=0.1,
        chosen_params_path="chosen.json",
        walkforward_path="wf.parquet",
        provenance="test",
    )


def test_allocation_never_assigns_to_quarantined_strategy() -> None:
    weights = allocate_capital(
        {
            "baseline": _state("baseline", GovernanceState.LIVE),
            "trend": _state("trend", GovernanceState.QUARANTINED),
        },
        evidence_by_slug={"baseline": _evidence("baseline", 0.5), "trend": _evidence("trend", 2.0)},
        config=AllocationConfig(mode="dsr-weighted"),
    )

    assert weights == {"baseline": 1.0}


def test_equal_live_uses_cap_and_renormalizes() -> None:
    weights = allocate_capital(
        {
            "a": _state("a", GovernanceState.LIVE),
            "b": _state("b", GovernanceState.LIVE),
            "c": _state("c", GovernanceState.LIVE),
        },
        evidence_by_slug={},
        config=AllocationConfig(mode="equal-live", max_weight=0.40),
    )

    assert weights == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_dsr_weighted_prefers_stronger_evidence() -> None:
    weights = allocate_capital(
        {
            "a": _state("a", GovernanceState.LIVE),
            "b": _state("b", GovernanceState.LIVE),
        },
        evidence_by_slug={"a": _evidence("a", 0.4), "b": _evidence("b", 0.8)},
        config=AllocationConfig(mode="dsr-weighted", max_weight=0.80),
    )

    assert weights["b"] > weights["a"]
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def test_dsr_weighted_respects_minimum_for_live_strategies_when_feasible() -> None:
    weights = allocate_capital(
        {
            "a": _state("a", GovernanceState.LIVE),
            "b": _state("b", GovernanceState.LIVE),
            "c": _state("c", GovernanceState.LIVE),
        },
        evidence_by_slug={
            "a": _evidence("a", 10.0),
            "b": _evidence("b", 0.01),
            "c": _evidence("c", 0.01),
        },
        config=AllocationConfig(mode="dsr-weighted", max_weight=0.90, min_weight=0.05),
    )

    assert weights["b"] >= 0.05
    assert weights["c"] >= 0.05
    assert abs(sum(weights.values()) - 1.0) < 1e-12


# --- risk-based allocation: pure core -----------------------------------------


def _rng_returns(mean: float, std: float, n: int = 252, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


def test_strategy_risk_basic_mean_std() -> None:
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.0] * 20)  # 100 obs
    mean, std = strategy_risk(r, min_observations=60)
    assert mean == pytest_approx(float(r.mean()))
    assert std == pytest_approx(float(r.std(ddof=1)))


def test_strategy_risk_nan_when_too_few_observations() -> None:
    mean, std = strategy_risk(np.array([0.01, 0.02, -0.01]), min_observations=60)
    assert math.isnan(mean) and math.isnan(std)


def test_strategy_risk_nan_when_zero_vol() -> None:
    _mean, std = strategy_risk(np.zeros(100), min_observations=60)
    assert math.isnan(std)


def test_risk_parity_favors_lower_vol() -> None:
    returns = {
        "calm": _rng_returns(0.0005, 0.005, seed=1),
        "wild": _rng_returns(0.0005, 0.020, seed=2),
    }
    raw = risk_based_raw_weights(
        returns, ["calm", "wild"], "risk-parity", AllocationConfig(mode="risk-parity")
    )
    assert raw is not None
    assert raw["calm"] > raw["wild"]


def test_fractional_kelly_favors_higher_edge_and_zeroes_negative() -> None:
    returns = {
        "edge": _rng_returns(0.0015, 0.010, seed=3),
        "noedge": _rng_returns(-0.0015, 0.010, seed=4),
    }
    raw = risk_based_raw_weights(
        returns, ["edge", "noedge"], "fractional-kelly", AllocationConfig(mode="fractional-kelly")
    )
    assert raw is None or raw["edge"] > raw["noedge"]
    # A purely-negative-mean strategy contributes 0 Kelly weight.
    only_neg = risk_based_raw_weights(
        {"noedge": _rng_returns(-0.002, 0.01, seed=5)},
        ["noedge"],
        "fractional-kelly",
        AllocationConfig(mode="fractional-kelly"),
    )
    assert only_neg is None  # all-zero raw -> caller falls back


def test_risk_based_returns_none_when_a_live_slug_lacks_curve() -> None:
    raw = risk_based_raw_weights(
        {"a": _rng_returns(0.001, 0.01, seed=6)},  # 'b' missing
        ["a", "b"],
        "risk-parity",
        AllocationConfig(mode="risk-parity"),
    )
    assert raw is None


# --- risk-based allocation: through allocate_capital --------------------------


def test_allocate_risk_parity_end_to_end() -> None:
    states = {
        "calm": _state("calm", GovernanceState.LIVE),
        "wild": _state("wild", GovernanceState.LIVE),
    }
    returns = {
        "calm": _rng_returns(0.0005, 0.005, seed=7),
        "wild": _rng_returns(0.0005, 0.020, seed=8),
    }
    weights = allocate_capital(
        states,
        evidence_by_slug={},
        config=AllocationConfig(mode="risk-parity", max_weight=0.90, min_weight=0.0),
        returns_by_slug=returns,
    )
    assert weights["calm"] > weights["wild"]
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def test_allocate_risk_mode_falls_back_to_equal_live_on_missing_data() -> None:
    states = {
        "a": _state("a", GovernanceState.LIVE),
        "b": _state("b", GovernanceState.LIVE),
        "c": _state("c", GovernanceState.LIVE),
    }
    # No returns at all -> must reproduce equal-live exactly.
    weights = allocate_capital(
        states,
        evidence_by_slug={},
        config=AllocationConfig(mode="risk-parity"),
        returns_by_slug={},
    )
    assert weights == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_allocate_risk_mode_respects_cap_and_floor() -> None:
    states = {s: _state(s, GovernanceState.LIVE) for s in ("a", "b", "c")}
    returns = {
        "a": _rng_returns(0.0005, 0.002, seed=9),  # very low vol -> would dominate
        "b": _rng_returns(0.0005, 0.020, seed=10),
        "c": _rng_returns(0.0005, 0.020, seed=11),
    }
    weights = allocate_capital(
        states,
        evidence_by_slug={},
        config=AllocationConfig(mode="risk-parity", max_weight=0.40, min_weight=0.05),
        returns_by_slug=returns,
    )
    assert max(weights.values()) <= 0.40 + 1e-9
    assert min(weights.values()) >= 0.05 - 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def test_single_live_strategy_is_full_weight_for_risk_mode() -> None:
    weights = allocate_capital(
        {"solo": _state("solo", GovernanceState.LIVE)},
        evidence_by_slug={},
        config=AllocationConfig(mode="fractional-kelly"),
        returns_by_slug={"solo": _rng_returns(0.001, 0.01, seed=12)},
    )
    assert weights == {"solo": 1.0}


# --- loader -------------------------------------------------------------------


def test_load_strategy_returns_reads_curve_and_skips_missing(tmp_path: Path) -> None:
    root = tmp_path
    wf_dir = root / "data" / "backtests" / "a"
    wf_dir.mkdir(parents=True)
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    equity = pd.Series(100_000.0 * (1.0 + 0.001) ** np.arange(10), index=idx, name="equity")
    equity.to_frame().rename_axis("timestamp").to_parquet(wf_dir / "walkforward.parquet")

    ev_a = _evidence("a", 0.5)
    object.__setattr__(ev_a, "walkforward_path", "data/backtests/a/walkforward.parquet")
    ev_b = _evidence("b", 0.5)
    object.__setattr__(ev_b, "walkforward_path", "data/backtests/b/walkforward.parquet")  # missing

    out = load_strategy_returns({"a": ev_a, "b": ev_b}, root=root)
    assert "a" in out and "b" not in out
    assert len(out["a"]) == 9  # pct_change drops the first row


def pytest_approx(x: float) -> object:
    import pytest

    return pytest.approx(x, rel=1e-9, abs=1e-12)


# --- covariance-aware HRP allocation ------------------------------------------


def test_hrp_two_strategies_reduce_to_inverse_variance() -> None:
    # For two strategies HRP's recursive bisection reduces to inverse-VARIANCE weights:
    # w_a = var_b / (var_a + var_b). Verify against the closed form on the sample cov.
    a = _rng_returns(0.0005, 0.005, seed=11)
    b = _rng_returns(0.0005, 0.020, seed=12)
    raw = hrp_raw_weights({"a": a, "b": b}, ["a", "b"], AllocationConfig(mode="hrp"))
    assert raw is not None
    cov = pd.DataFrame({"a": a, "b": b}).cov()  # same path hrp_raw_weights uses internally
    expected_a = float(cov.iloc[1, 1] / (cov.iloc[0, 0] + cov.iloc[1, 1]))
    assert raw["a"] == pytest_approx(expected_a)
    assert raw["b"] == pytest_approx(1.0 - expected_a)
    assert raw["a"] > raw["b"]  # lower-vol strategy gets the larger weight
    assert abs(raw["a"] + raw["b"] - 1.0) < 1e-12


def test_hrp_three_strategies_valid_simplex_and_deterministic() -> None:
    returns = {
        "x": _rng_returns(0.0005, 0.008, seed=21),
        "y": _rng_returns(0.0005, 0.012, seed=22),
        "z": _rng_returns(0.0005, 0.016, seed=23),
    }
    cfg = AllocationConfig(mode="hrp")
    raw1 = hrp_raw_weights(returns, ["x", "y", "z"], cfg)
    raw2 = hrp_raw_weights(returns, ["x", "y", "z"], cfg)
    assert raw1 is not None and raw2 is not None
    assert set(raw1) == {"x", "y", "z"}
    assert all(w > 0.0 for w in raw1.values())
    assert abs(sum(raw1.values()) - 1.0) < 1e-9
    assert raw1 == raw2  # deterministic


def test_hrp_returns_none_when_a_live_slug_lacks_curve() -> None:
    raw = hrp_raw_weights(
        {"a": _rng_returns(0.001, 0.01, seed=31)},  # 'b' missing
        ["a", "b"],
        AllocationConfig(mode="hrp"),
    )
    assert raw is None


def test_hrp_returns_none_on_insufficient_observations() -> None:
    raw = hrp_raw_weights(
        {
            "a": _rng_returns(0.001, 0.01, n=30, seed=32),
            "b": _rng_returns(0.001, 0.01, n=30, seed=33),
        },
        ["a", "b"],
        AllocationConfig(mode="hrp", min_observations=60),
    )
    assert raw is None


def test_hrp_returns_none_on_degenerate_flat_curve() -> None:
    raw = hrp_raw_weights(
        {"a": _rng_returns(0.001, 0.01, seed=34), "b": np.zeros(252)},  # flat -> zero vol
        ["a", "b"],
        AllocationConfig(mode="hrp"),
    )
    assert raw is None


def test_allocate_hrp_end_to_end() -> None:
    states = {
        "calm": _state("calm", GovernanceState.LIVE),
        "wild": _state("wild", GovernanceState.LIVE),
    }
    returns = {
        "calm": _rng_returns(0.0005, 0.005, seed=41),
        "wild": _rng_returns(0.0005, 0.020, seed=42),
    }
    weights = allocate_capital(
        states,
        evidence_by_slug={},
        config=AllocationConfig(mode="hrp", max_weight=0.90, min_weight=0.0),
        returns_by_slug=returns,
    )
    assert weights["calm"] > weights["wild"]  # diversifies toward the lower-vol strategy
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def test_allocate_hrp_falls_back_to_equal_live_on_missing_data() -> None:
    states = {
        "a": _state("a", GovernanceState.LIVE),
        "b": _state("b", GovernanceState.LIVE),
    }
    weights = allocate_capital(
        states,
        evidence_by_slug={},
        config=AllocationConfig(mode="hrp", max_weight=0.90, min_weight=0.0),
        returns_by_slug=None,  # no curves -> fail open to equal-live
    )
    assert weights["a"] == pytest_approx(0.5)
    assert weights["b"] == pytest_approx(0.5)
