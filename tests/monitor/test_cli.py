from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

import quant.cli as cli_mod
from quant.cli import cli
from quant.governance.halt import load_halt
from quant.monitor.status import read_status


def _write_equity(data_dir: Path, values: list[float]) -> None:
    live = data_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=len(values)), "equity": values})
    df.to_parquet(live / "equity.parquet")


def test_guard_check_prints_and_never_halts(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_equity(tmp_data_dir, [90, 95, 100, 100, 100, 85, 70])  # deep drawdown
    monkeypatch.setattr(cli_mod, "_best_effort_positions", lambda settings: (None, "skipped"))
    res = CliRunner().invoke(cli, ["guard", "check"])
    assert res.exit_code == 0, res.output
    assert "account_drawdown" in res.output
    # check must NOT halt and must NOT write status
    assert load_halt(tmp_data_dir).active is False
    assert read_status(tmp_data_dir) is None


def test_guard_run_once_dry_run_writes_status_no_halt(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_equity(tmp_data_dir, [90, 95, 100, 100, 100, 85, 70])
    monkeypatch.setattr(cli_mod, "_best_effort_positions", lambda settings: (None, "skipped"))
    res = CliRunner().invoke(cli, ["guard", "run", "--once", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert load_halt(tmp_data_dir).active is False
    status = read_status(tmp_data_dir)
    assert status is not None and status.worst_severity == "halt"


def test_guard_run_once_auto_halts_on_breach(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_equity(tmp_data_dir, [90, 95, 100, 100, 100, 85, 70])
    monkeypatch.setattr(cli_mod, "_best_effort_positions", lambda settings: (None, "skipped"))
    res = CliRunner().invoke(cli, ["guard", "run", "--once"])
    assert res.exit_code == 0, res.output
    assert load_halt(tmp_data_dir).active is True
    assert "auto-halt" in load_halt(tmp_data_dir).reason
    assert "resume" in res.output.lower()
