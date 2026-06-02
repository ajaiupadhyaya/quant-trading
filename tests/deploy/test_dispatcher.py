"""Dispatcher tick: pure decision wired to injected runner/clock/alerts/lock."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant.deploy.dispatcher import Dispatcher
from quant.deploy.manifest import load_manifest
from quant.deploy.markers import read_markers

MANIFEST = Path(__file__).resolve().parents[2] / "quant" / "deploy" / "jobs.toml"


class _Runner:
    def __init__(self, rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.rc = rc

    def __call__(self, args: list[str], cwd: Path) -> int:
        self.calls.append(args)
        return self.rc


class _Alerts:
    def __init__(self) -> None:
        self.success: list[str | None] = []
        self.emergencies: list[tuple[str, str]] = []

    def ping_success(self, url: str | None) -> None:
        self.success.append(url)

    def ping_fail(self, url: str | None, body: str = "") -> None:
        pass

    def send_emergency(self, title: str, message: str) -> bool:
        self.emergencies.append((title, message))
        return True


def _disp(tmp_path: Path, runner: _Runner, alerts: _Alerts) -> Dispatcher:
    return Dispatcher(
        data_dir=tmp_path,
        manifest=load_manifest(MANIFEST),
        runner=runner,
        alerts=alerts,
        halt_active=lambda: False,
    )


def test_fresh_premarket_runs_chain_and_marks(tmp_path: Path) -> None:
    runner, alerts = _Runner(), _Alerts()
    # Tue 2026-06-02 09:01 ET = 13:01 UTC (EDT)
    rc = _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 13, 1, tzinfo=UTC))
    assert rc == 0
    assert ["data", "refresh", "--start", "2018-01-01"] in runner.calls
    assert ["rebalance", "--dry-run"] in runner.calls
    assert read_markers(tmp_path).get("premarket-health") is not None
    assert alerts.success  # clean tick pinged liveness


def test_failed_chain_writes_no_marker_and_suppresses_success_ping(tmp_path: Path) -> None:
    runner, alerts = _Runner(rc=1), _Alerts()
    _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 13, 1, tzinfo=UTC))
    assert read_markers(tmp_path).get("premarket-health") is None  # no marker on failure
    assert alerts.success == []  # a failed job suppresses the success heartbeat


def test_doctor_failure_stops_rebalance_chain(tmp_path: Path) -> None:
    # A runner that fails ONLY on `doctor` must stop the daily-rebalance chain
    # before `rebalance` runs.
    class _DoctorFails(_Runner):
        def __call__(self, args: list[str], cwd: Path) -> int:
            self.calls.append(args)
            return 1 if args == ["doctor"] else 0

    runner, alerts = _DoctorFails(), _Alerts()
    # 2026-06-02 15:56 ET = 19:56 UTC -> rebalance FRESH window
    _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 19, 56, tzinfo=UTC))
    assert ["doctor"] in runner.calls
    assert ["rebalance"] not in runner.calls  # chain stopped at doctor


def test_missed_critical_fires_emergency_not_trade(tmp_path: Path) -> None:
    runner, alerts = _Runner(), _Alerts()
    # 2026-06-02 16:05 ET = 20:05 UTC -> past rebalance hard cutoff
    _disp(tmp_path, runner, alerts).tick(now_utc=datetime(2026, 6, 2, 20, 5, tzinfo=UTC))
    assert ["rebalance"] not in runner.calls
    assert any("rebalance" in m.lower() for _t, m in alerts.emergencies)


def test_halt_active_fires_emergency_and_runs_no_trade(tmp_path: Path) -> None:
    runner, alerts = _Runner(), _Alerts()
    d = Dispatcher(
        data_dir=tmp_path,
        manifest=load_manifest(MANIFEST),
        runner=runner,
        alerts=alerts,
        halt_active=lambda: True,
    )
    d.tick(now_utc=datetime(2026, 6, 2, 19, 56, tzinfo=UTC))
    assert ["rebalance"] not in runner.calls
    assert alerts.emergencies  # fresh halt pushed
