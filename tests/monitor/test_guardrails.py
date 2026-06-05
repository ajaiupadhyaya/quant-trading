from __future__ import annotations

from quant.governance.drift import DriftRow
from quant.live.safety import CheckResult, StrategyRiskBudget
from quant.monitor.guardrails import (
    GuardrailConfig,
    GuardrailInputs,
    evaluate_account_drawdown,
    evaluate_bar_freshness,
    evaluate_drift,
    evaluate_equity_health,
    evaluate_guardrails,
    evaluate_reconciliation,
)


def _row(flag: str, window: int = 20, z: float = -2.5) -> DriftRow:
    return DriftRow(
        strategy="account",
        window=window,
        realized_return=-0.1,
        expected_return=0.0,
        z_score=z,
        flag=flag,  # type: ignore[arg-type]
    )


def test_drift_halt_on_halt_candidate() -> None:
    out = evaluate_drift([_row("normal"), _row("halt_candidate")])
    assert out.severity == "halt"
    assert out.name == "drift"


def test_drift_warn_on_watch_only() -> None:
    out = evaluate_drift([_row("normal"), _row("watch")])
    assert out.severity == "warn"


def test_drift_ok_when_normal_or_empty() -> None:
    assert evaluate_drift([_row("normal")]).severity == "ok"
    assert evaluate_drift([]).severity == "ok"


def test_account_drawdown_halt_on_breach() -> None:
    budget = StrategyRiskBudget(max_drawdown=0.25)
    assert evaluate_account_drawdown(-0.30, budget).severity == "halt"
    assert evaluate_account_drawdown(-0.25, budget).severity == "halt"  # at threshold
    assert evaluate_account_drawdown(-0.10, budget).severity == "ok"
    assert evaluate_account_drawdown(0.0, budget).severity == "ok"


def test_equity_health_ok_when_positive() -> None:
    out = evaluate_equity_health(1_000_000.0, source="live", had_positive_history=False)
    assert out.severity == "ok"
    assert out.name == "equity_health"
    assert (
        evaluate_equity_health(50_000.0, source="local", had_positive_history=True).severity == "ok"
    )


def test_equity_health_halts_on_live_zero() -> None:
    # A live account reporting $0 is a real wipeout / dead feed, never "ok".
    out = evaluate_equity_health(0.0, source="live", had_positive_history=False)
    assert out.severity == "halt"
    assert evaluate_equity_health(-5.0, source="live", had_positive_history=True).severity == "halt"


def test_equity_health_halts_on_collapse_after_positive_history() -> None:
    out = evaluate_equity_health(0.0, source="local", had_positive_history=True)
    assert out.severity == "halt"


def test_equity_health_warns_on_pure_data_gap() -> None:
    # No source / never-positive local series: surfaced as warn, but never halts
    # (fail-open on a monitoring gap is preserved).
    assert evaluate_equity_health(0.0, source="none", had_positive_history=False).severity == "warn"
    assert (
        evaluate_equity_health(0.0, source="local", had_positive_history=False).severity == "warn"
    )


def test_reconciliation_severity() -> None:
    assert evaluate_reconciliation(None, halt_on_breach=False).severity == "ok"
    ok = CheckResult(ok=True, name="reconciliation", detail="fine")
    assert evaluate_reconciliation(ok, halt_on_breach=False).severity == "ok"
    bad = CheckResult(ok=False, name="reconciliation", detail="diff")
    assert evaluate_reconciliation(bad, halt_on_breach=False).severity == "warn"
    assert evaluate_reconciliation(bad, halt_on_breach=True).severity == "halt"


def test_bar_freshness_severity() -> None:
    assert evaluate_bar_freshness(None).severity == "ok"
    ok = CheckResult(ok=True, name="bar_freshness", detail="fresh")
    assert evaluate_bar_freshness(ok).severity == "ok"
    stale = CheckResult(ok=False, name="bar_freshness", detail="stale")
    assert evaluate_bar_freshness(stale).severity == "warn"


def test_evaluate_guardrails_aggregates_worst() -> None:
    inputs = GuardrailInputs(
        drift_rows=[_row("halt_candidate")],
        account_drawdown_pct=-0.05,
        latest_equity=100_000.0,
        reconciliation=CheckResult(ok=True, name="reconciliation", detail=""),
        bar_freshness=CheckResult(ok=False, name="bar_freshness", detail="stale"),
    )
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert {o.name for o in report.outcomes} == {
        "drift",
        "account_drawdown",
        "equity_health",
        "reconciliation",
        "bar_freshness",
    }
    assert report.worst_severity == "halt"
    assert report.halting is True


def test_evaluate_guardrails_all_ok() -> None:
    inputs = GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=0.0,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )
    report = evaluate_guardrails(inputs, GuardrailConfig())
    assert report.worst_severity == "ok"
    assert report.halting is False
