"""Phase 8 — macro-conditioned regime HMM + Bayesian online change-point detector."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant.forecast.regime import (
    BOCPDConfig,
    MacroRegimeConfig,
    RegimeMetrics,
    _verdict,
    bocpd_run_length,
    build_macro_regime_features,
    change_point_series,
    compare_regime_models,
    compute_change_points,
    compute_macro_regime,
    live_macro_regime,
    render_macro_regime,
)
from quant.regime.detect import DetectConfig


def _series(seed: int, n: int = 800, vol: float = 0.01, start: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    px = start * np.exp(np.cumsum(vol * rng.standard_normal(n)))
    return pd.Series(px, index=idx)


def _macro_inputs(n: int = 800) -> dict[str, pd.Series]:
    """Synthetic SPY + macro series sharing a business-day index."""
    spy = _series(0, n)
    idx = spy.index
    rng = np.random.default_rng(1)
    return {
        "spy_close": spy,
        "vix": pd.Series(18 + 4 * rng.standard_normal(n).cumsum() / np.sqrt(n), index=idx).clip(
            9, 80
        ),
        "dgs10": pd.Series(3.0 + 0.2 * rng.standard_normal(n).cumsum() / np.sqrt(n), index=idx),
        "dgs2": pd.Series(2.5 + 0.2 * rng.standard_normal(n).cumsum() / np.sqrt(n), index=idx),
        "baa": pd.Series(5.0 + 0.3 * rng.standard_normal(n).cumsum() / np.sqrt(n), index=idx),
        "aaa": pd.Series(4.0 + 0.2 * rng.standard_normal(n).cumsum() / np.sqrt(n), index=idx),
    }


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
def test_macro_conditioning_adds_a_credit_dimension() -> None:
    inp = _macro_inputs()
    market = build_macro_regime_features(
        spy_close=inp["spy_close"],
        vix=inp["vix"],
        dgs10=inp["dgs10"],
        dgs2=inp["dgs2"],
        baa=None,
        aaa=None,
        macro_config=MacroRegimeConfig(use_credit=False),
    )
    macro = build_macro_regime_features(
        spy_close=inp["spy_close"],
        vix=inp["vix"],
        dgs10=inp["dgs10"],
        dgs2=inp["dgs2"],
        baa=inp["baa"],
        aaa=inp["aaa"],
        macro_config=MacroRegimeConfig(use_credit=True),
    )
    assert "credit" not in market.columns
    assert "credit" in macro.columns
    assert macro.shape[1] == market.shape[1] + 1
    # standardized → roughly mean-zero, unit-scale (rolling), never NaN after dropna
    assert macro["credit"].notna().all()
    assert abs(float(macro["credit"].mean())) < 2.0


# --------------------------------------------------------------------------- #
# BOCPD
# --------------------------------------------------------------------------- #
def test_bocpd_run_length_rows_are_normalized() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    rl = bocpd_run_length(x, BOCPDConfig(max_run=50))
    assert rl.shape == (200, 51)
    assert np.allclose(rl.sum(axis=1), 1.0, atol=1e-6)


def test_bocpd_detects_a_structural_break() -> None:
    rng = np.random.default_rng(7)
    # 250 calm days, then a sharp variance + mean shift for 100 days.
    calm = 0.5 * rng.standard_normal(250)
    shock = 3.0 + 3.0 * rng.standard_normal(100)
    x = np.concatenate([calm, shock])
    rl = bocpd_run_length(x, BOCPDConfig(hazard_lambda=250, short_run=5))
    cp = rl[:, :6].sum(axis=1)
    exp_run = (rl * np.arange(rl.shape[1])).sum(axis=1)
    # Change-point mass jumps right after the break and the expected run length
    # collapses, versus the calm stretch just before it.
    assert cp[250:270].max() > 5 * cp[200:245].mean()
    assert exp_run[245] > exp_run[255]  # run length resets at the break


def test_change_point_series_burns_warmup_and_reads_tail() -> None:
    ret = pd.Series(
        np.log(_series(5, 700).astype(float)).diff().to_numpy(), index=_series(5, 700).index
    ).dropna()
    df = change_point_series(ret, BOCPDConfig())
    assert list(df.columns) == ["cp_prob", "exp_run"]
    assert len(df) < len(ret)  # leading burn-in dropped
    assert (df["cp_prob"].between(0.0, 1.0)).all()


def test_compute_change_points_failopen_on_short_series() -> None:
    short = pd.Series([0.001, -0.002, 0.0], index=pd.bdate_range("2020-01-01", periods=3))
    r = compute_change_points(short)
    assert r.cp_prob is None  # too short → degraded, never raises


# --------------------------------------------------------------------------- #
# A/B comparison + verdict
# --------------------------------------------------------------------------- #
def test_compare_regime_models_runs_and_scores_both() -> None:
    inp = _macro_inputs(900)
    cmp = compare_regime_models(
        spy_close=inp["spy_close"],
        vix=inp["vix"],
        dgs10=inp["dgs10"],
        dgs2=inp["dgs2"],
        baa=inp["baa"],
        aaa=inp["aaa"],
        detect_config=DetectConfig(train_window_days=252, n_restarts=2),
    )
    assert cmp.market.n_features == 5
    assert cmp.macro.n_features == 6  # +credit
    assert cmp.market.n > 0 and cmp.macro.n > 0
    assert isinstance(cmp.verdict, str) and cmp.verdict


def test_verdict_thresholds() -> None:
    def mk(dd_red: float) -> RegimeMetrics:
        return RegimeMetrics("x", 100, 6, None, None, None, None, -0.3, -0.25, dd_red)

    assert "HELPS" in _verdict(mk(-0.05), mk(-0.04), 0.05)
    assert "HURTS" in _verdict(mk(-0.04), mk(-0.05), -0.05)
    assert "marginal" in _verdict(mk(-0.04), mk(-0.04), 0.005)
    assert "inconclusive" in _verdict(mk(-0.04), mk(-0.04), None)


# --------------------------------------------------------------------------- #
# Live read + render
# --------------------------------------------------------------------------- #
def test_compute_macro_regime_labels_today() -> None:
    inp = _macro_inputs(700)
    feats = build_macro_regime_features(
        spy_close=inp["spy_close"],
        vix=inp["vix"],
        dgs10=inp["dgs10"],
        dgs2=inp["dgs2"],
        baa=inp["baa"],
        aaa=inp["aaa"],
    )
    r = compute_macro_regime(
        feats,
        feats.index[-1].date(),
        detect_config=DetectConfig(train_window_days=252, n_restarts=2),
    )
    assert r.label in ("calm-bull", "choppy", "crisis")
    assert r.n_features == 6
    assert abs((r.p_calm or 0) + (r.p_choppy or 0) + (r.p_crisis or 0) - 1.0) < 1e-6
    assert r.credit_z is not None


def test_live_macro_regime_failopen(tmp_path, monkeypatch) -> None:
    # Force the data load to fail → degraded read, never raises (fail-open).
    import quant.forecast.regime as reg

    def _boom(*a, **k):
        raise RuntimeError("no data")

    monkeypatch.setattr(reg, "_load_macro_inputs", _boom)
    r = live_macro_regime(SimpleNamespace(data_dir=tmp_path), date(2026, 6, 4))
    assert r.label is None and r.n_features == 0


def test_render_macro_regime() -> None:
    assert render_macro_regime(None) == "Macro regime: unavailable"
    from quant.forecast.regime import MacroRegimeRead

    r = MacroRegimeRead(
        asof="2026-06-04",
        label="calm-bull",
        p_calm=0.7,
        p_choppy=0.2,
        p_crisis=0.1,
        n_features=6,
        credit_z=-0.4,
        cp_prob=0.05,
        oos_verdict="marginal",
    )
    out = render_macro_regime(r)
    assert "calm-bull" in out and "credit_z" in out and "cp_prob" in out
