"""Tests for the Click CLI scaffold."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from quant.cli import cli


def test_cli_help_succeeds() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "quant" in result.output.lower()


@pytest.mark.parametrize(
    "subcommand",
    ["status", "backtest", "validate", "rebalance", "tearsheet", "journal", "monitor", "data"],
)
def test_cli_subcommand_help_succeeds(subcommand: str) -> None:
    result = CliRunner().invoke(cli, [subcommand, "--help"])
    assert result.exit_code == 0, result.output


def test_cli_backtest_unknown_strategy_errors(fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["backtest", "definitely-not-a-strategy"])
    assert result.exit_code != 0
    assert (
        "unknown strategy" in result.output.lower() or "definitely-not-a-strategy" in result.output
    )


def test_cli_status_renders(fake_env: None) -> None:
    mock_alpaca = MagicMock()
    mock_alpaca.account.return_value = MagicMock(
        equity=100000.0,
        last_equity=99500.0,
        buying_power=50000.0,
        cash=25000.0,
        portfolio_value=100000.0,
        pattern_day_trader=False,
    )
    mock_alpaca.positions.return_value = []
    with patch("quant.cli.AlpacaClient", return_value=mock_alpaca):
        result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "100000" in result.output or "100,000" in result.output


def test_cli_data_inventory_runs(fake_env: None, tmp_data_dir) -> None:
    result = CliRunner().invoke(cli, ["data", "inventory"])
    assert result.exit_code == 0
    assert "universe" in result.output
