"""Autonomous monitoring daemon + kill-switch — the headless guardian (pillar 2).

The daemon evaluates guardrails each tick and auto-pulls the existing
kill-switch on a halt verdict. It can HALT but never resumes (resume is always
a manual `quant governance resume`).
"""

from quant.monitor.daemon import (
    TickResult,
    format_heartbeat,
    gather_inputs,
    run_loop,
    run_once,
)
from quant.monitor.guardrails import (
    GuardrailConfig,
    GuardrailInputs,
    GuardrailOutcome,
    GuardrailReport,
    Severity,
    evaluate_account_drawdown,
    evaluate_bar_freshness,
    evaluate_drift,
    evaluate_equity_health,
    evaluate_guardrails,
    evaluate_reconciliation,
)
from quant.monitor.status import MonitorStatus, monitor_status_path, read_status, write_status

__all__ = [
    "GuardrailConfig",
    "GuardrailInputs",
    "GuardrailOutcome",
    "GuardrailReport",
    "MonitorStatus",
    "Severity",
    "TickResult",
    "evaluate_account_drawdown",
    "evaluate_bar_freshness",
    "evaluate_drift",
    "evaluate_equity_health",
    "evaluate_guardrails",
    "evaluate_reconciliation",
    "format_heartbeat",
    "gather_inputs",
    "monitor_status_path",
    "read_status",
    "run_loop",
    "run_once",
    "write_status",
]
