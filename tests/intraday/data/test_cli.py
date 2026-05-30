# tests/intraday/data/test_cli.py
from click.testing import CliRunner

from quant.cli import cli


def test_intraday_data_status_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(cli, ["intraday", "data", "status"])
    assert result.exit_code == 0
    assert "Intraday data" in result.output


def test_intraday_data_doctor_reports_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(cli, ["intraday", "data", "doctor"])
    assert result.exit_code == 0
    assert "partitions" in result.output.lower()
