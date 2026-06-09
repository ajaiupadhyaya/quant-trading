from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_mm_group_exists():
    r = CliRunner().invoke(intraday, ["mm", "--help"])
    assert r.exit_code == 0
    assert "simulate" in r.output and "sweep" in r.output


def test_simulate_prints_pnl_and_inventory():
    r = CliRunner().invoke(intraday, ["mm", "simulate", "--symbol", "QQQ", "--seed", "5"])
    assert r.exit_code == 0
    assert "QQQ" in r.output
    assert "pnl" in r.output.lower()
    assert "inventory" in r.output.lower()


def test_sweep_prints_table_with_gamma_and_note():
    r = CliRunner().invoke(intraday, ["mm", "sweep", "--symbol", "QQQ"])
    assert r.exit_code == 0
    assert "gamma" in r.output.lower()
    assert "stylized" in r.output.lower()
