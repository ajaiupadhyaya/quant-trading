from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.regimes import (
    REGIMES,
    RegimeBreakdown,
    Regime,
    compute_regime_breakdown,
    count_positive_regimes,
)


def test_regimes_constant_has_all_five_windows() -> None:
    slugs = {r.slug for r in REGIMES}
    assert slugs == {"gfc-2008", "china-2015", "covid-2020", "bear-2022", "bull-2024"}


def test_each_regime_has_start_before_end() -> None:
    for r in REGIMES:
        assert r.start < r.end, r.slug


def test_compute_regime_breakdown_returns_one_entry_per_regime() -> None:
    idx = pd.bdate_range("2005-01-01", "2025-01-01")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    breakdown = compute_regime_breakdown(returns)
    assert len(breakdown) == len(REGIMES)
    slugs = [b.slug for b in breakdown]
    assert slugs == [r.slug for r in REGIMES]


def test_breakdown_handles_regime_with_no_overlap() -> None:
    # Only 2024 dates → 2008 GFC should be empty.
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    returns = pd.Series(0.001, index=idx)
    breakdown = compute_regime_breakdown(returns)
    gfc = next(b for b in breakdown if b.slug == "gfc-2008")
    assert gfc.n_days == 0
    assert gfc.total_return == 0.0
    assert gfc.sharpe == 0.0


def test_count_positive_regimes_matches_total_return_signs() -> None:
    breakdown = [
        RegimeBreakdown(
            slug=f"r{i}",
            name=f"R{i}",
            start=pd.Timestamp("2020-01-01").date(),
            end=pd.Timestamp("2020-06-01").date(),
            n_days=100,
            total_return=tr,
            sharpe=0.0,
            max_drawdown=0.0,
        )
        for i, tr in enumerate([0.1, -0.05, 0.02, -0.01, 0.0])
    ]
    # Strictly positive only: 0.1 and 0.02 → 2
    assert count_positive_regimes(breakdown) == 2


def test_breakdown_with_constant_positive_drift_yields_positive_total_return() -> None:
    idx = pd.bdate_range("2008-01-01", "2009-06-01")
    returns = pd.Series(0.001, index=idx)
    breakdown = compute_regime_breakdown(returns)
    gfc = next(b for b in breakdown if b.slug == "gfc-2008")
    assert gfc.n_days > 0
    assert gfc.total_return > 0
