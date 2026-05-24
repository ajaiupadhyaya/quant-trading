"""Tests for the Click CLI scaffold."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from quant.cli import cli
from quant.strategies import REGISTRY
from quant.strategies.base import Strategy, StrategySpec


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


class _CLIToyStrategy(Strategy):
    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="cli-toy",
        name="CLI Toy",
        description="-",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAA": 1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAA": 1}

    @classmethod
    def build(cls, bars: pd.DataFrame, params: dict[str, object] | None = None) -> Strategy:
        return cls(params=params)


def test_data_refresh_command(tmp_data_dir: Path, fake_env: None) -> None:
    runner = CliRunner()
    with patch("quant.cli.refresh_caches") as mock_refresh:
        mock_refresh.return_value = type(
            "R",
            (),
            {
                "symbols": ["SPY"],
                "symbols_fetched": 1,
                "rows_total": 5,
                "elapsed_s": 0.1,
                "errors": [],
            },
        )()
        result = runner.invoke(cli, ["data", "refresh"])
    assert result.exit_code == 0, result.output
    assert "symbols_fetched" in result.output.lower() or "fetched" in result.output.lower()


def test_backtest_command_unknown_strategy(fake_env: None) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest", "definitely-not-a-strategy"])
    assert result.exit_code != 0
    assert "unknown strategy" in result.output.lower()


def test_backtest_command_runs_registered_strategy(
    tmp_data_dir: Path,
    fake_env: None,
) -> None:
    REGISTRY["cli-toy"] = _CLIToyStrategy
    try:
        runner = CliRunner()
        with patch("quant.data.bars._fetch_alpaca") as mock_alpaca:
            # Provide enough synthetic data via the bars fetch mock.
            dates = pd.bdate_range("2010-01-01", "2024-12-31")
            df = pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1,
                },
                index=pd.DatetimeIndex(dates, name="timestamp"),
            )
            mock_alpaca.return_value = {"AAA": df, "BBB": df}
            result = runner.invoke(cli, ["backtest", "cli-toy", "--quick"])
        assert result.exit_code == 0, result.output
        # Check the tear-sheet was written:
        out_dir = tmp_data_dir / "backtests" / "cli-toy"
        assert (out_dir / "tearsheet.html").exists()
    finally:
        REGISTRY.pop("cli-toy", None)


def test_tearsheet_command_opens_html(tmp_data_dir: Path, fake_env: None) -> None:
    # Pre-create a fake tear-sheet
    out_dir = tmp_data_dir / "backtests" / "stub"
    out_dir.mkdir(parents=True)
    (out_dir / "tearsheet.html").write_text("<html></html>")

    runner = CliRunner()
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(cli, ["tearsheet", "stub"])
    assert result.exit_code == 0, result.output
    mock_open.assert_called_once()


def test_tearsheet_command_missing_file(tmp_data_dir: Path, fake_env: None) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["tearsheet", "nonexistent"])
    assert result.exit_code != 0
    assert "tearsheet" in result.output.lower()


def test_validate_command_exit_code_when_strategy_unknown():
    from click.testing import CliRunner
    from quant.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "no-such-strategy"])
    assert result.exit_code != 0


def test_validate_command_runs_to_completion_on_known_strategy(monkeypatch, tmp_path, fake_env, tmp_data_dir):
    """Smoke: validate completes (pass or fail) and writes a tear-sheet."""
    from datetime import date

    import pandas as pd
    from click.testing import CliRunner

    from quant.cli import cli
    from quant.strategies import REGISTRY
    from quant.strategies.base import Strategy, StrategySpec
    from tests.conftest import synthetic_bars

    class _Smoke(Strategy):
        spec = StrategySpec(
            slug="cli-smoke",
            name="CLI Smoke",
            description="",
            universe=["AAA"],
            rebalance_frequency="monthly",
        )

        def generate_signals(self, asof):
            return pd.Series({"AAA": 1.0})

        def target_positions(self, asof, equity):
            return {"AAA": 10}

    REGISTRY["cli-smoke"] = _Smoke
    try:
        bars = synthetic_bars(["AAA"], date(2010, 1, 1), date(2020, 12, 31))
        monkeypatch.setattr("quant.cli.get_bars", lambda _req: bars)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["validate", "cli-smoke", "--start", "2010-01-01", "--end", "2020-12-31",
             "--bootstrap-resamples", "50"],
        )
        # exit_code may be 0 (pass) or 2 (fail-gate); both indicate the command ran.
        assert result.exit_code in (0, 2), result.output
        assert "Deflated Sharpe" in result.output
    finally:
        del REGISTRY["cli-smoke"]
