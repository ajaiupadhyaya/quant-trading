"""The guard loop pings its OWN healthcheck each successful tick (decoupled
from the dispatcher), so guard-death is independently observable."""

from __future__ import annotations

from pathlib import Path

from quant.monitor.daemon import run_loop
from quant.monitor.guardrails import GuardrailConfig, GuardrailInputs


def _clean_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.01,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def test_run_loop_pings_healthcheck_each_tick(tmp_path: Path) -> None:
    pings: list[str] = []
    inputs = _clean_inputs()  # empty inputs -> all guardrails OK
    run_loop(
        tmp_path,
        GuardrailConfig(),
        interval_s=0.0,
        max_ticks=2,
        inputs_fn=lambda: inputs,
        sleep=lambda _s: None,
        heartbeat_ping=lambda: pings.append("ok"),
    )
    assert pings == ["ok", "ok"]
