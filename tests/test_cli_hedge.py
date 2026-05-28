from click.testing import CliRunner

from quant.cli import cli


def test_hedge_price_prints_greeks():
    runner = CliRunner()
    res = runner.invoke(
        cli, ["hedge", "price", "--spot", "500", "--strike", "480", "--days", "30", "--vol", "0.2"]
    )
    assert res.exit_code == 0, res.output
    assert "delta" in res.output.lower()
    assert "price" in res.output.lower()


def test_hedge_price_with_mark_shows_iv():
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "hedge",
            "price",
            "--spot",
            "500",
            "--strike",
            "480",
            "--days",
            "30",
            "--vol",
            "0.2",
            "--mark",
            "8.0",
            "--right",
            "put",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "implied" in res.output.lower()
