from __future__ import annotations

import numpy as np
import pandas as pd

from quant.regime.validation import RegimeReport, validate_regime_series


def _regime_frame(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n)
    # Persistent blocks: 200 calm, 40 crisis, repeating.
    labels = []
    while len(labels) < n:
        labels += ["calm-bull"] * 200 + ["crisis"] * 40
    labels = labels[:n]
    p = {"calm-bull": (0.8, 0.1, 0.1), "choppy": (0.1, 0.8, 0.1), "crisis": (0.1, 0.1, 0.8)}
    frame = pd.DataFrame(
        {
            "p_calm": [p[lbl][0] for lbl in labels],
            "p_choppy": [p[lbl][1] for lbl in labels],
            "p_crisis": [p[lbl][2] for lbl in labels],
            "label": labels,
            "refit_epoch": 0,
        },
        index=idx,
    )
    frame.index.name = "date"
    return frame


def test_validate_returns_report_with_four_gates():
    frame = _regime_frame(600)
    rng = np.random.default_rng(1)
    # Returns that crash during crisis labels — so de-risking helps.
    rets = pd.Series(
        np.where(frame["label"].to_numpy() == "crisis", -0.02, 0.001)
        + rng.normal(0, 0.005, len(frame)),
        index=frame.index,
    )
    report = validate_regime_series(frame, spy_returns=rets)
    assert isinstance(report, RegimeReport)
    assert set(report.gates) == {
        "persistence",
        "coherence",
        "predictive_lift",
        "pit_consistent",
    }
    # Crisis returns are clearly worse, so de-risking lifts the drawdown metric.
    assert report.gates["predictive_lift"] is True
    assert report.gates["persistence"] is True


def test_churny_series_fails_persistence():
    idx = pd.bdate_range("2018-01-01", periods=300)
    labels = np.array(["calm-bull", "crisis"] * 150)  # flips every day
    frame = pd.DataFrame(
        {
            "p_calm": 0.5,
            "p_choppy": 0.0,
            "p_crisis": 0.5,
            "label": labels,
            "refit_epoch": 0,
        },
        index=idx,
    )
    rets = pd.Series(np.zeros(300), index=idx)
    report = validate_regime_series(frame, spy_returns=rets)
    assert report.gates["persistence"] is False
