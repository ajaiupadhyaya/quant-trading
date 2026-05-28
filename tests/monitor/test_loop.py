from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.governance.halt import load_halt
from quant.monitor.daemon import run_loop
from quant.monitor.guardrails import GuardrailConfig, GuardrailInputs
from quant.util.logging import logger

NOW = datetime(2026, 5, 28, 14, 0, 0, tzinfo=UTC)


def _clean_inputs() -> GuardrailInputs:
    return GuardrailInputs(
        drift_rows=[],
        account_drawdown_pct=-0.01,
        latest_equity=100_000.0,
        reconciliation=None,
        bar_freshness=None,
    )


def test_run_loop_runs_max_ticks_without_real_sleep(tmp_path: Path) -> None:
    slept: list[float] = []
    printed: list[str] = []
    results = run_loop(
        tmp_path,
        GuardrailConfig(),
        interval_s=999.0,
        max_ticks=3,
        inputs_fn=lambda: _clean_inputs(),
        sleep=lambda s: slept.append(s),
        console_print=printed.append,
        now_fn=lambda: NOW,
    )
    assert len(results) == 3
    assert len(printed) == 3
    # sleeps only happen BETWEEN ticks, not after the last
    assert len(slept) == 2
    assert load_halt(tmp_path).active is False


def test_run_loop_continues_on_tick_error(tmp_path: Path) -> None:
    printed: list[str] = []
    calls = {"n": 0}

    def flaky_inputs() -> GuardrailInputs:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _clean_inputs()

    results = run_loop(
        tmp_path,
        GuardrailConfig(),
        interval_s=0.0,
        max_ticks=2,
        inputs_fn=flaky_inputs,
        sleep=lambda s: None,
        console_print=printed.append,
        now_fn=lambda: NOW,
    )
    # first tick errored (no TickResult), second succeeded
    assert len(results) == 1
    assert any("error" in line.lower() for line in printed)


def test_run_loop_logs_tick_error_without_console_sink(tmp_path: Path) -> None:
    # A headless daemon (console_print=None) must still surface tick errors via
    # the logger rather than failing silently.
    def boom() -> GuardrailInputs:
        raise RuntimeError("transient")

    logged: list[str] = []
    handle = logger.add(lambda m: logged.append(str(m)), level="WARNING")
    try:
        results = run_loop(
            tmp_path,
            GuardrailConfig(),
            interval_s=0.0,
            max_ticks=1,
            inputs_fn=boom,
            sleep=lambda s: None,
            console_print=None,
            now_fn=lambda: NOW,
        )
    finally:
        logger.remove(handle)
    assert results == []
    assert any("monitor tick error" in line for line in logged)
