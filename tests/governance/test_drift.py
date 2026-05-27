"""Tests for paper P&L drift flags."""

from __future__ import annotations

import pandas as pd

from quant.governance.drift import DriftConfig, drift_flag, summarize_drift


def test_drift_flag_thresholds() -> None:
    assert drift_flag(0.5, DriftConfig()) == "normal"
    assert drift_flag(-1.5, DriftConfig()) == "watch"
    assert drift_flag(-2.5, DriftConfig()) == "halt_candidate"


def test_summarize_drift_flags_underperformance() -> None:
    dates = pd.bdate_range("2026-01-01", periods=30)
    paper = pd.Series([0.0] + [-0.01] * 29, index=dates)
    expected = pd.Series([0.001] * 30, index=dates)

    rows = summarize_drift({"baseline": paper}, {"baseline": expected}, windows=(20,))

    assert rows[0].strategy == "baseline"
    assert rows[0].window == 20
    assert rows[0].flag in {"watch", "halt_candidate"}
