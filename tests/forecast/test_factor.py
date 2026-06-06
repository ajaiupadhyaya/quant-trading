"""Phase 8 — cross-sectional factor model + purged OOS IC machinery."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.forecast.factor import (
    FACTOR_UNIVERSE,
    FactorConfig,
    build_factor_panel,
    composite_score,
    compute_factor_scores,
    cross_sectional_ic,
    fit_ridge,
    forward_returns,
    live_factor_scores,
    render_factor_scores,
    walk_forward_factor_eval,
    winsorized_zscore,
)


@pytest.fixture(autouse=True)
def _no_edgar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep factor tests offline: stub EDGAR so only price factors compute."""
    import quant.data.edgar as edgar

    monkeypatch.setattr(edgar, "fetch_company_facts", lambda *a, **k: pd.DataFrame())


def _panel(seed: int, n_days: int = 1500, n_names: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n_days)
    cols = [f"N{i}" for i in range(n_names)]
    steps = 0.01 * rng.standard_normal((n_days, n_names))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)


def test_winsorized_zscore() -> None:
    z = winsorized_zscore(pd.Series([1.0, 2, 3, 4, 5]), z=3.0)
    assert abs(z.mean()) < 1e-9
    # an outlier is clipped to ±z
    z2 = winsorized_zscore(pd.Series([0.0, 0, 0, 0, 100]), z=2.0)
    assert z2.max() <= 2.0 + 1e-9
    # constant input → all NaN
    assert winsorized_zscore(pd.Series([5.0, 5, 5])).isna().all()


def test_cross_sectional_ic_detects_correlation() -> None:
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    pear, rank = cross_sectional_ic(s, s * 2.0)  # perfectly aligned
    assert pear is not None and pear > 0.999
    assert rank is not None and rank > 0.999
    pear_neg, _ = cross_sectional_ic(s, -s)
    assert pear_neg is not None and pear_neg < -0.999
    assert cross_sectional_ic(pd.Series([1.0, 2]), pd.Series([1.0, 2])) == (None, None)


def test_composite_score_requires_min_factors() -> None:
    panel = pd.DataFrame(
        {
            "momentum": [1.0, 2, 3, 4],
            "value": [4.0, 3, 2, 1],
            "quality": [np.nan, np.nan, np.nan, np.nan],
        },
        index=["A", "B", "C", "D"],
    )
    score = composite_score(panel, config=FactorConfig(min_factors=2))
    assert score.notna().all()  # 2 factors present
    # momentum and value are opposite → composite ≈ 0 for symmetric inputs
    assert abs(score.mean()) < 1e-9


def test_fit_ridge_recovers_linear_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3))
    true = np.array([1.0, -0.5, 0.25])
    y = x @ true
    coef = fit_ridge(x, y, lam=1e-6)
    assert np.allclose(coef, true, atol=1e-2)


def test_forward_returns() -> None:
    closes = _panel(1, n_days=300, n_names=4)
    fr = forward_returns(closes, 100, 21)
    expected = closes.iloc[121] / closes.iloc[100] - 1.0
    assert np.allclose(fr.to_numpy(), expected.to_numpy())
    assert forward_returns(closes, 295, 21).isna().all()  # runs off the end


def test_build_factor_panel_price_factors(tmp_path) -> None:
    closes = _panel(2)
    panel = build_factor_panel(closes, len(closes) - 1, data_dir=tmp_path)
    assert list(panel.columns) == list(
        ("momentum", "low_vol", "reversal", "value", "quality", "investment")
    )
    assert panel["momentum"].notna().any()  # price factors present
    assert panel["value"].isna().all()  # EDGAR stubbed → fundamentals NaN


def test_walk_forward_eval_runs_and_reports_factors(tmp_path) -> None:
    closes = _panel(3, n_days=1800, n_names=30)
    ev = walk_forward_factor_eval(closes, data_dir=tmp_path, model="composite")
    assert ev.n_periods > 10
    assert "momentum" in ev.per_factor_ic  # price factors evaluated
    assert ev.hit_rate is None or 0.0 <= ev.hit_rate <= 1.0


def test_walk_forward_eval_ridge_runs(tmp_path) -> None:
    closes = _panel(4, n_days=1800, n_names=30)
    ev = walk_forward_factor_eval(closes, data_dir=tmp_path, model="ridge")
    assert ev.model == "ridge" and ev.n_periods > 5


def test_compute_factor_scores(tmp_path) -> None:
    closes = _panel(5)
    f = compute_factor_scores(closes, date(2020, 11, 2), data_dir=tmp_path, top_n=3)
    assert f.n_names > 10
    assert len(f.top) == 3 and len(f.bottom) == 3
    # top scores should exceed bottom scores
    assert f.scores[f.top[0]] > f.scores[f.bottom[0]]


def test_live_factor_scores_failopen(tmp_path) -> None:
    from types import SimpleNamespace

    f = live_factor_scores(SimpleNamespace(data_dir=tmp_path), date(2026, 6, 4))
    assert f.n_names == 0  # empty cache → degraded, never raises


def test_render() -> None:
    assert render_factor_scores(None) == "Factor model: unavailable"
    closes = _panel(6)
    f = compute_factor_scores(
        closes, date(2020, 11, 2), data_dir=__import__("pathlib").Path("/nonexistent")
    )
    out = render_factor_scores(f)
    assert "Factor model:" in out and "names" in out


def test_universe_is_operating_companies() -> None:
    assert len(FACTOR_UNIVERSE) == 49
    for etf in ("SPY", "GLD", "TLT", "EEM", "IEF", "DBC", "VNQ", "EFA"):
        assert etf not in FACTOR_UNIVERSE


# --- GBM model branch + DSR-gated research verdict ----------------------------


def test_gbm_walk_forward_runs_and_populates_spread_series() -> None:
    from quant.forecast.factor import walk_forward_factor_eval
    from quant.forecast.gbm import GBMConfig

    closes = _panel(seed=11, n_days=1500, n_names=30)
    cfg = FactorConfig(gbm=GBMConfig(n_estimators=30, max_depth=2, min_samples_leaf=5))
    ev = walk_forward_factor_eval(closes, data_dir=None, config=cfg, model="gbm")
    assert ev.model == "gbm"
    assert ev.n_periods > 0
    # The OOS long-short series is retained for DSR/PSR.
    assert len(ev.oos_spread_returns) > 0
    assert all(np.isfinite(s) for s in ev.oos_spread_returns)


def test_composite_eval_unchanged_by_gbm_addition() -> None:
    """Adding the gbm branch must not perturb the composite path (regression)."""
    from quant.forecast.factor import walk_forward_factor_eval

    closes = _panel(seed=12, n_days=1400, n_names=25)
    ev = walk_forward_factor_eval(closes, data_dir=None, model="composite")
    assert ev.model == "composite"
    assert ev.n_periods > 0
    # composite is parameter-free OOS; mean_tertile_spread is defined when periods exist
    assert ev.mean_tertile_spread is not None


def test_gbm_research_verdict_computes_dsr_psr_and_gate() -> None:
    from quant.forecast.factor import GBMVerdict, gbm_research_verdict
    from quant.forecast.gbm import GBMConfig

    closes = _panel(seed=13, n_days=1600, n_names=30)
    cfg = FactorConfig(gbm=GBMConfig(n_estimators=25, max_depth=2))
    v = gbm_research_verdict(closes, data_dir=None, config=cfg)
    assert isinstance(v, GBMVerdict)
    assert v.n_periods > 0
    # On pure random-walk closes there is no real edge, so DSR/PSR should NOT pass
    # the live bar — the honest, expected outcome the charter anticipates.
    assert v.deflated_sharpe is not None
    assert v.probabilistic_sharpe is not None
    assert v.passes is (v.passes_dsr and v.passes_psr)
    assert not v.passes  # no edge in noise -> not promotion-eligible
