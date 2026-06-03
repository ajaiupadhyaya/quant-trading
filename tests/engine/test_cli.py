"""`quant engine` CLI: run --once / state / status (offline, no broker)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from quant.cli import cli
from quant.engine import loop as lp
from quant.engine.loop import engine_dir


@pytest.fixture(autouse=True)
def _no_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never touch Alpaca in offline CLI tests.
    monkeypatch.setattr(lp, "_default_positions", lambda settings: None)
    monkeypatch.setattr(lp, "_default_equity", lambda settings: None)


def test_engine_run_once_dry_run_exit_zero(tmp_data_dir: Path, fake_env: object) -> None:
    res = CliRunner().invoke(cli, ["engine", "run", "--once", "--dry-run"])
    assert res.exit_code == 0
    assert (engine_dir(tmp_data_dir) / "state.json").exists()


def test_engine_state_and_status(tmp_data_dir: Path, fake_env: object) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["engine", "run", "--once", "--dry-run"])
    r_state = runner.invoke(cli, ["engine", "state"])
    assert r_state.exit_code == 0
    assert "session_phase" in r_state.output
    r_status = runner.invoke(cli, ["engine", "status"])
    assert r_status.exit_code == 0
    assert "heartbeat" in r_status.output


def test_engine_state_errors_when_absent(tmp_data_dir: Path, fake_env: object) -> None:
    res = CliRunner().invoke(cli, ["engine", "state"])
    assert res.exit_code != 0  # no run yet -> clear error
