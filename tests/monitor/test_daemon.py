from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from quant.governance.drift import DriftRow
from quant.governance.halt import load_halt, set_halt
from quant.live.safety import CheckResult
from quant.monitor.daemon import format_heartbeat, gather_inputs, run_once
from quant.monitor.guardrails import GuardrailConfig, GuardrailInputs, evaluate_guardrails
from quant.monitor.status import read_status

NOW = datetime(2026, 5, 28, 14, 0, 0, tzinfo=UTC)


def _halt_row() -> DriftRow:
    return DriftRow(
        strategy="account",
        window=20,
        realized_return=-0.2,
        expected_return=0.0,
        z_score=-3.0,
        flag="halt_candidate",
    )


def _clean_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.02,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def _drawdown_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.30,
        latest_equity=70_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def test_run_once_clean_no_halt(tmp_path: Path) -> None:
    res = run_once(tmp_path, GuardrailConfig(), inputs=_clean_inputs(), now=NOW)
    assert res.halt_triggered is False
    assert res.halt_active is False
    assert res.report.worst_severity == "ok"
    assert load_halt(tmp_path).active is False
    status = read_status(tmp_path)
    assert status is not None and status.worst_severity == "ok"


def test_run_once_drawdown_breach_halts(tmp_path: Path) -> None:
    res = run_once(tmp_path, GuardrailConfig(), inputs=_drawdown_inputs(), now=NOW)
    assert res.halt_triggered is True
    assert res.halt_active is True
    halt = load_halt(tmp_path)
    assert halt.active is True
    assert "account_drawdown" in halt.reason


def test_run_once_drift_halt_candidate_halts(tmp_path: Path) -> None:
    inputs = GuardrailInputs(
        drift_rows=[_halt_row()],
        account_drawdown_pct=-0.01,
        latest_equity=99_000.0,
        reconciliation=None,
        bar_freshness=None,
    )
    res = run_once(tmp_path, GuardrailConfig(), inputs=inputs, now=NOW)
    assert res.halt_triggered is True
    assert "drift" in load_halt(tmp_path).reason


def test_run_once_idempotent_when_already_halted(tmp_path: Path) -> None:
    set_halt(tmp_path, reason="manual: original reason", created_at=NOW)
    res = run_once(tmp_path, GuardrailConfig(), inputs=_drawdown_inputs(), now=NOW)
    assert res.halt_triggered is False  # did not re-halt
    assert res.halt_active is True
    # original reason preserved
    assert load_halt(tmp_path).reason == "manual: original reason"


def test_run_once_dry_run_does_not_halt(tmp_path: Path) -> None:
    res = run_once(tmp_path, GuardrailConfig(), inputs=_drawdown_inputs(), now=NOW, dry_run=True)
    assert res.halt_triggered is False
    assert load_halt(tmp_path).active is False  # NOT halted
    status = read_status(tmp_path)
    assert status is not None and status.worst_severity == "halt"  # but recorded


def test_run_once_warn_only_does_not_halt(tmp_path: Path) -> None:
    inputs = GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.05,
        latest_equity=100_000.0,
        reconciliation=CheckResult(ok=False, name="reconciliation", detail="diff"),
        bar_freshness=CheckResult(ok=False, name="bar_freshness", detail="stale"),
    )
    res = run_once(tmp_path, GuardrailConfig(), inputs=inputs, now=NOW)
    assert res.report.worst_severity == "warn"
    assert res.halt_triggered is False
    assert load_halt(tmp_path).active is False


def test_gather_inputs_missing_equity_warns_but_never_false_halts(tmp_path: Path) -> None:
    # No equity.parquet and no live feed -> a pure monitoring GAP: surfaced as
    # equity_health=warn (no longer silently "ok"), but never a false halt.
    inputs = gather_inputs(tmp_path, asof=date(2026, 5, 28), config=GuardrailConfig())
    assert inputs.drift_rows == []
    assert inputs.account_drawdown_pct == 0.0
    assert inputs.equity_source == "none"
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert report.halting is False
    equity_health = next(o for o in report.outcomes if o.name == "equity_health")
    assert equity_health.severity == "warn"


def test_gather_inputs_live_equity_takes_precedence(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir(parents=True)
    df = pd.DataFrame(
        {"date": pd.bdate_range("2026-01-01", periods=3), "equity": [100.0, 100.0, 100.0]}
    )
    df.to_parquet(live / "equity.parquet")
    inputs = gather_inputs(
        tmp_path, asof=date(2026, 5, 28), config=GuardrailConfig(), live_equity=2_000_000.0
    )
    assert inputs.latest_equity == 2_000_000.0
    assert inputs.equity_source == "live"
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert report.halting is False


def test_run_once_live_zero_equity_halts(tmp_path: Path) -> None:
    # A live account reporting $0 is a wipeout / dead feed -> must HALT, not "ok".
    res = run_once(tmp_path, GuardrailConfig(), live_equity=0.0, now=NOW)
    assert res.halt_triggered is True
    assert "equity_health" in load_halt(tmp_path).reason


def test_gather_inputs_reads_equity_and_computes_drawdown(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir(parents=True)
    # ramp to 100 then crash to 70 -> ~ -30% drawdown
    equity = [90.0, 95.0, 100.0, 100.0, 100.0, 100.0, 85.0, 70.0]
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=len(equity)), "equity": equity})
    df.to_parquet(live / "equity.parquet")
    inputs = gather_inputs(tmp_path, asof=date(2026, 5, 28), config=GuardrailConfig())
    assert inputs.account_drawdown_pct <= -0.29
    assert inputs.latest_equity == 70.0


def test_format_heartbeat_marks_halt() -> None:
    inputs = _drawdown_inputs()
    report = evaluate_guardrails(inputs, GuardrailConfig())
    line = format_heartbeat(inputs, report, NOW, halt_active=True)
    assert "HALT" in line
    assert "dd" in line
