import pytest

pytest.importorskip("torch")  # skip cleanly where torch is absent

from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_dl_train_runs():
    runner = CliRunner()
    result = runner.invoke(intraday, ["dl", "train", "--epochs", "5", "--n", "800", "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "loss" in result.output.lower()


def test_dl_evaluate_runs_both_tracks():
    runner = CliRunner()
    result = runner.invoke(
        intraday, ["dl", "evaluate", "--epochs", "5", "--n", "800", "--seed", "7"]
    )
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "synthetic" in out and "random" in out
    assert "lstm" in out and "naive" in out and "linear" in out
    # The honesty note (EMH) must be printed.
    assert "emh" in out or "near-unforecastable" in out or "does not beat" in out
