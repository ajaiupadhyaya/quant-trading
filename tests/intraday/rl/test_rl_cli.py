from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_rl_group_exists():
    r = CliRunner().invoke(intraday, ["rl", "--help"])
    assert r.exit_code == 0
    assert "train" in r.output and "compare" in r.output


def test_train_prints_convergence():
    r = CliRunner().invoke(intraday, ["rl", "train", "--shares", "20", "--episodes", "3000"])
    assert r.exit_code == 0
    assert "converg" in r.output.lower() or "cost" in r.output.lower()


def test_compare_prints_three_policies_and_note():
    r = CliRunner().invoke(intraday, ["rl", "compare", "--shares", "20", "--episodes", "3000"])
    assert r.exit_code == 0
    out = r.output.lower()
    assert "learned" in out and "twap" in out and ("almgren" in out or "a-c" in out)
    assert "rediscover" in out or "stylized" in out
