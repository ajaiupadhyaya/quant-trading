from __future__ import annotations

from quant.monitor import (
    GuardrailConfig,
    GuardrailInputs,
    GuardrailOutcome,
    GuardrailReport,
    MonitorStatus,
    TickResult,
    evaluate_guardrails,
    gather_inputs,
    monitor_status_path,
    read_status,
    run_loop,
    run_once,
)


def test_public_api_importable() -> None:
    assert GuardrailConfig and GuardrailInputs and GuardrailOutcome and GuardrailReport
    assert MonitorStatus and TickResult
    assert evaluate_guardrails and gather_inputs and run_once and run_loop
    assert monitor_status_path and read_status
