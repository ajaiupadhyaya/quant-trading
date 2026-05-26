"""Targeted tests for ``RiskParity`` parameter contract.

General smoke tests live in ``test_concrete_strategies.py`` and the LW
shrinkage estimator gets its own coverage in ``test_ledoit_wolf.py``. This
file locks in the widened param grid + ``shrinkage_floor`` contract added
in Task 1.4 of the 2026-05-25 go-live plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.risk_parity import ledoit_wolf_shrinkage


def test_risk_parity_default_params_include_shrinkage_floor() -> None:
    from quant.strategies.risk_parity import RiskParity

    assert "shrinkage_floor" in RiskParity.default_params


def test_risk_parity_param_grid_widened() -> None:
    """Locks in the widened grid contract — controls future drift."""
    from quant.strategies.risk_parity import RiskParity

    grid = RiskParity.param_grid
    assert 0.06 in grid["vol_target_annual"]
    assert 0.12 in grid["vol_target_annual"]
    assert 63 in grid["lookback_days"]
    assert 504 in grid["lookback_days"]
    assert "shrinkage_floor" in grid
    assert 0.0 in grid["shrinkage_floor"]
    assert 0.40 in grid["shrinkage_floor"]


def test_ledoit_wolf_floor_raises_delta_when_below() -> None:
    """A floor above the natural ``delta`` must pin ``delta`` to the floor."""
    rng = np.random.default_rng(0)
    n, k, corr = 200, 5, 0.2
    base = rng.normal(0, 0.01, size=(n, 1))
    idio = rng.normal(0, 0.01, size=(n, k))
    rho = float(np.sqrt(corr))
    panel = pd.DataFrame(
        rho * base + float(np.sqrt(1 - corr)) * idio,
        columns=[f"A{i}" for i in range(k)],
    )
    _, delta_unfloored = ledoit_wolf_shrinkage(panel, floor=0.0)
    # This config produces a moderate natural delta (~0.54). Use a floor a
    # solid margin above it so the assertion is robust to small RNG drift.
    floor = min(delta_unfloored + 0.20, 0.99)
    _, delta_floored = ledoit_wolf_shrinkage(panel, floor=floor)
    assert delta_floored >= floor - 1e-12
    assert delta_floored > delta_unfloored


def test_ledoit_wolf_floor_does_not_lower_natural_delta() -> None:
    """A floor below the closed-form ``delta`` must leave ``delta`` unchanged."""
    rng = np.random.default_rng(11)
    # Small n / large k => high natural shrinkage.
    panel = pd.DataFrame(
        rng.normal(0, 0.01, size=(30, 12)),
        columns=[f"A{i}" for i in range(12)],
    )
    _, delta_unfloored = ledoit_wolf_shrinkage(panel, floor=0.0)
    _, delta_floored = ledoit_wolf_shrinkage(panel, floor=0.05)
    # The natural delta should be above the tiny floor we passed.
    assert delta_unfloored >= 0.05
    np.testing.assert_allclose(delta_floored, delta_unfloored, rtol=1e-12, atol=1e-12)
