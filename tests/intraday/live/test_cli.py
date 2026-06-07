from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_live_group_exists():
    r = CliRunner().invoke(intraday, ["live", "--help"])
    assert r.exit_code == 0
    for cmd in ("run", "status", "halt", "resume", "flat"):
        assert cmd in r.output


def test_halt_then_status_then_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    assert runner.invoke(intraday, ["live", "halt", "--reason", "test"]).exit_code == 0
    out = runner.invoke(intraday, ["live", "status"]).output
    assert "HALTED" in out.upper()
    assert runner.invoke(intraday, ["live", "resume", "--reason", "ok"]).exit_code == 0
