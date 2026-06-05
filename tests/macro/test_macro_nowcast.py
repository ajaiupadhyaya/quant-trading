"""Roadmap track C — macro / business-cycle nowcast."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from quant.macro.nowcast import (
    MacroNowcastConfig,
    compute_macro_nowcast,
    live_macro_nowcast,
    render_macro_nowcast,
)

ASOF = date(2026, 6, 4)


def test_expansion_when_curve_positive_credit_tight() -> None:
    n = compute_macro_nowcast(
        ASOF,
        t10y3m=1.20,
        t10y2y=0.80,
        baa=5.5,
        aaa=4.7,
        hy_oas=3.0,
        nfci=-0.5,
        claims=210_000,
        claims_year_low=205_000,
        breakeven10=2.3,
        sahm=0.10,
    )
    assert n.recession_signal is False
    assert n.recession_risk_label == "low"
    assert n.cycle_label == "expansion"
    assert abs(n.credit_spread_baa_aaa - 0.8) < 1e-9
    # 5 sub-scores: curve, hy_oas, nfci, claims, sahm (baa/aaa only form the spread)
    assert n.n_components == 5


def test_late_cycle_on_inversion() -> None:
    n = compute_macro_nowcast(
        ASOF,
        t10y3m=-0.40,  # inverted
        hy_oas=4.5,
        nfci=0.1,
        claims=230_000,
        claims_year_low=210_000,
        sahm=0.20,
    )
    assert n.cycle_label == "late-cycle"  # inverted curve
    assert n.recession_signal is False


def test_contraction_when_sahm_triggers() -> None:
    n = compute_macro_nowcast(ASOF, t10y3m=-0.5, hy_oas=7.0, nfci=0.8, sahm=0.60)
    assert n.recession_signal is True
    assert n.cycle_label == "contraction"
    assert n.recession_risk_label == "high"


def test_recession_risk_rises_with_stress() -> None:
    calm = compute_macro_nowcast(ASOF, t10y3m=1.0, hy_oas=3.0, nfci=-0.5, sahm=0.0)
    stressed = compute_macro_nowcast(ASOF, t10y3m=-1.0, hy_oas=8.0, nfci=1.0, sahm=0.5)
    assert calm.recession_risk is not None and stressed.recession_risk is not None
    assert stressed.recession_risk > calm.recession_risk
    assert calm.recession_risk_label == "low"
    assert stressed.recession_risk_label == "high"


def test_claims_vs_year_low() -> None:
    n = compute_macro_nowcast(ASOF, t10y3m=0.5, claims=250_000, claims_year_low=200_000)
    assert abs(n.claims_vs_year_low - 0.25) < 1e-9


def test_low_coverage_suppresses_labels() -> None:
    n = compute_macro_nowcast(ASOF, hy_oas=4.0)  # one component only
    assert n.n_components == 1
    assert n.recession_risk is not None  # still computed
    assert n.recession_risk_label is None  # below min_components
    assert n.cycle_label is None


def test_empty_inputs_never_raise() -> None:
    n = compute_macro_nowcast(ASOF)
    assert n.n_components == 0
    assert n.recession_risk is None
    assert n.recession_signal is False
    assert n.cycle_label is None
    assert n.credit_spread_baa_aaa is None


def test_nonfinite_inputs_sanitized() -> None:
    n = compute_macro_nowcast(ASOF, t10y3m=float("nan"), hy_oas=4.0, nfci=-0.2)
    assert n.term_spread_10y3m is None  # NaN dropped
    assert n.n_components == 2  # hy_oas + nfci


def test_live_macro_nowcast_failopen(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant.data import macro as macro_mod

    monkeypatch.setattr(macro_mod, "get_series", lambda code: (_ for _ in ()).throw(RuntimeError()))
    n = live_macro_nowcast(SimpleNamespace(), ASOF, config=MacroNowcastConfig())
    assert n.recession_risk is None and n.n_components == 0
    assert n.cycle_label is None  # FRED down → degraded but no exception


def test_live_macro_nowcast_computes_claims_year_low(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    def fake_get_series(code: str) -> pd.Series:
        if code == "ICSA":
            return pd.Series([200_000, 210_000, 260_000])  # last 260k, low 200k → +30%
        raise RuntimeError("only claims stubbed")

    monkeypatch.setattr("quant.data.macro.get_series", fake_get_series)
    n = live_macro_nowcast(SimpleNamespace(), ASOF)
    assert n.initial_claims == 260_000
    assert abs(n.claims_vs_year_low - 0.30) < 1e-9


def test_render() -> None:
    assert render_macro_nowcast(None) == "Macro nowcast: unavailable"
    n = compute_macro_nowcast(
        ASOF, t10y3m=-0.4, hy_oas=4.5, nfci=0.1, claims=230_000, claims_year_low=210_000, sahm=0.2
    )
    out = render_macro_nowcast(n)
    assert "cycle=late-cycle" in out
    assert "HY_OAS=4.5%" in out
    assert "10y3m=-0.40" in out


def test_n_components_counts_subscores() -> None:
    # curve + hy_oas + nfci + claims + sahm = 5 sub-scores
    n = compute_macro_nowcast(
        ASOF, t10y3m=1.0, hy_oas=3.0, nfci=-0.5, claims=210_000, claims_year_low=205_000, sahm=0.1
    )
    assert n.n_components == 5
