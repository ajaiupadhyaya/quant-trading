"""Tests for the Click CLI scaffold."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from quant.cli import _validation_command, cli
from quant.strategies import REGISTRY
from quant.strategies.base import Strategy, StrategySpec


def test_cli_help_succeeds() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "quant" in result.output.lower()


@pytest.mark.parametrize(
    "subcommand",
    [
        "status",
        "backtest",
        "validate",
        "rebalance",
        "tearsheet",
        "journal",
        "monitor",
        "data",
        "research",
        "risk",
    ],
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


def test_research_cli_lists_and_compares_experiments(tmp_data_dir: Path, fake_env: None) -> None:
    from datetime import UTC, datetime

    from quant.research.registry import ExperimentRecord, append_experiment

    path = tmp_data_dir / "research" / "experiments.jsonl"
    append_experiment(
        path,
        ExperimentRecord(
            run_id="a",
            created_at=datetime(2026, 5, 28, tzinfo=UTC),
            strategy="trend",
            kind="validation",
            git_sha="abc",
            command="quant validate trend",
            params={},
            metrics={"dsr": 0.3},
            gates={"overall": True},
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=1.0,
        ),
    )
    append_experiment(
        path,
        ExperimentRecord(
            run_id="b",
            created_at=datetime(2026, 5, 28, tzinfo=UTC),
            strategy="momentum",
            kind="validation",
            git_sha="abc",
            command="quant validate momentum",
            params={},
            metrics={"dsr": 0.7},
            gates={"overall": True},
            artifacts={},
            data_snapshot_id=None,
            wall_time_seconds=1.0,
        ),
    )

    runner = CliRunner()
    listed = runner.invoke(cli, ["research", "list"])
    compared = runner.invoke(cli, ["research", "compare", "a", "b"])
    ranked = runner.invoke(cli, ["research", "leaderboard", "--metric", "dsr"])

    assert listed.exit_code == 0, listed.output
    assert "trend" in listed.output
    assert compared.exit_code == 0, compared.output
    assert "+0.4000" in compared.output
    assert ranked.exit_code == 0, ranked.output
    assert ranked.output.find("│ b") < ranked.output.find("│ a")


def test_data_snapshot_and_quality_commands_write_artifacts(
    tmp_data_dir: Path, fake_env: None
) -> None:
    raw = tmp_data_dir / "raw" / "SPY.parquet"
    dates = pd.bdate_range("2026-01-01", periods=3)
    pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0],
            "close": [2.0, 3.0, 4.0],
            "volume": [100, 100, 100],
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    ).to_parquet(raw)

    runner = CliRunner()
    snapshot = runner.invoke(
        cli,
        [
            "data",
            "snapshot",
            "--symbols",
            "SPY",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-06",
            "--snapshot-id",
            "cli-snap",
        ],
    )
    quality = runner.invoke(
        cli,
        ["data", "quality", "--symbols", "SPY", "--start", "2026-01-01", "--end", "2026-01-06"],
    )

    assert snapshot.exit_code == 0, snapshot.output
    assert (tmp_data_dir / "snapshots" / "cli-snap" / "manifest.json").exists()
    assert quality.exit_code == 0, quality.output
    assert (tmp_data_dir / "ops" / "health" / "data_quality.json").exists()


def test_risk_pretrade_command_writes_report(tmp_data_dir: Path, fake_env: None) -> None:
    from quant.execution.orders import OrderSide, OrderTemplate
    from quant.live.rebalance import RebalanceReport, StrategyRebalanceOutcome

    planned = RebalanceReport(
        asof=date(2026, 5, 28),
        equity=100_000.0,
        enabled_strategies=["defensive-etf-allocation"],
        dry_run=True,
        outcomes=[
            StrategyRebalanceOutcome(
                slug="defensive-etf-allocation",
                target={"SPY": 20},
                previous={},
                orders=[
                    OrderTemplate(
                        symbol="SPY",
                        qty=20,
                        side=OrderSide.BUY,
                        strategy_slug="defensive-etf-allocation",
                    )
                ],
                reference_prices={"SPY": 500.0},
            )
        ],
    )

    with patch("quant.live.run_rebalance", return_value=planned) as run:
        result = CliRunner().invoke(cli, ["risk", "pretrade"])

    run.assert_called_once_with(
        dry_run=True,
        skip_safety_checks=True,
        record_bookkeeping=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_data_dir / "risk" / "pretrade_report.json").read_text())
    assert payload["passed"] is True
    assert payload["gross_exposure"] == 0.1
    assert payload["symbol_weights"] == {"SPY": 0.1}
    assert payload["rebalance"]["total_orders"] == 1


def test_doctor_uses_governance_live_strategies(tmp_data_dir: Path) -> None:
    from datetime import datetime

    from quant.cli import _doctor_governance_live_slugs
    from quant.governance.models import GovernanceState, StrategyState
    from quant.governance.store import strategy_states_path, write_strategy_states

    write_strategy_states(
        strategy_states_path(tmp_data_dir),
        {
            "defensive-etf-allocation": StrategyState(
                slug="defensive-etf-allocation",
                state=GovernanceState.LIVE,
                evaluated_at=datetime(2026, 5, 28),
                validation_age_days=0,
                reason_codes=[],
                reason="eligible",
                code_enabled_live=True,
            ),
            "momentum": StrategyState(
                slug="momentum",
                state=GovernanceState.QUARANTINED,
                evaluated_at=datetime(2026, 5, 28),
                validation_age_days=1,
                reason_codes=["gate_failed"],
                reason="bootstrap failed",
                code_enabled_live=True,
            ),
        },
    )

    slugs, error = _doctor_governance_live_slugs(tmp_data_dir)

    assert error is None
    assert slugs == ["defensive-etf-allocation"]


def test_data_quality_default_end_uses_last_completed_session() -> None:
    from quant.cli import _default_data_quality_end_date

    assert _default_data_quality_end_date(date(2026, 5, 28)) == date(2026, 5, 27)
    assert _default_data_quality_end_date(date(2026, 5, 30)) == date(2026, 5, 29)


def test_governance_halt_and_resume_commands(tmp_data_dir: Path, fake_env: None) -> None:
    runner = CliRunner()
    halted = runner.invoke(cli, ["governance", "halt", "--reason", "stop"])
    resumed = runner.invoke(cli, ["governance", "resume", "--reason", "healthy"])

    assert halted.exit_code == 0, halted.output
    assert "halted" in halted.output.lower()
    assert resumed.exit_code == 0, resumed.output
    assert "resumed" in resumed.output.lower()


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
        # Check the tear-sheet AND chosen_params.json were written:
        out_dir = tmp_data_dir / "backtests" / "cli-toy"
        assert (out_dir / "tearsheet.html").exists()
        params_path = out_dir / "chosen_params.json"
        assert params_path.exists()
        import json

        payload = json.loads(params_path.read_text())
        assert "latest" in payload
        assert "windows" in payload
        assert isinstance(payload["windows"], list)
    finally:
        REGISTRY.pop("cli-toy", None)


def test_combined_book_help_succeeds() -> None:
    result = CliRunner().invoke(cli, ["combined-book", "--help"])
    assert result.exit_code == 0
    assert "combined" in result.output.lower() or "all live-enabled" in result.output.lower()


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


def test_validation_command_records_quick_flag() -> None:
    command = _validation_command(
        strategy="pairs",
        start_date=date(2010, 1, 1),
        end_date=date(2026, 5, 26),
        bootstrap_resamples=1000,
        bootstrap_seed=0,
        quick=True,
    )

    assert command == (
        "quant validate pairs --start 2010-01-01 --end 2026-05-26 "
        "--bootstrap-resamples 1000 --bootstrap-seed 0 --quick"
    )


def test_validate_command_runs_to_completion_on_known_strategy(
    monkeypatch, tmp_path, fake_env, tmp_data_dir
):
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
            [
                "validate",
                "cli-smoke",
                "--start",
                "2010-01-01",
                "--end",
                "2020-12-31",
                "--bootstrap-resamples",
                "50",
            ],
        )
        # exit_code may be 0 (pass) or 2 (fail-gate); both indicate the command ran.
        assert result.exit_code in (0, 2), result.output
        assert "Deflated Sharpe" in result.output
        experiments = tmp_data_dir / "research" / "experiments.jsonl"
        assert experiments.exists()
        assert "cli-smoke" in experiments.read_text()
    finally:
        del REGISTRY["cli-smoke"]


def test_governance_help_succeeds(fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["governance", "--help"])
    assert result.exit_code == 0, result.output
    assert "status" in result.output
    assert "refresh" in result.output


def test_governance_refresh_writes_artifacts(tmp_data_dir: Path, fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["governance", "refresh", "--asof", "2026-05-26"])
    assert result.exit_code == 0, result.output
    assert (tmp_data_dir / "governance" / "validation_manifest.json").exists()
    assert (tmp_data_dir / "governance" / "strategy_states.json").exists()
    assert (tmp_data_dir / "governance" / "allocation.json").exists()


def test_governance_status_renders_unknown_when_artifacts_missing(
    tmp_data_dir: Path, fake_env: None
) -> None:
    result = CliRunner().invoke(cli, ["governance", "status"])
    assert result.exit_code == 0, result.output
    assert "unknown" in result.output.lower()


def test_governance_drift_writes_report(tmp_data_dir: Path, fake_env: None) -> None:
    from quant.live.bookkeeping import append_equity_row

    for i, asof in enumerate(pd.bdate_range("2026-01-01", periods=25).date):
        append_equity_row(
            tmp_data_dir,
            asof=asof,
            equity=100_000.0 + i * 100.0,
            last_equity=99_900.0 + i * 100.0,
            cash=10_000.0,
            buying_power=200_000.0,
            portfolio_value=100_000.0 + i * 100.0,
        )

    result = CliRunner().invoke(cli, ["governance", "drift"])
    assert result.exit_code == 0, result.output
    assert (tmp_data_dir / "governance" / "drift_report.json").exists()


def test_governance_audit_reports_missing_validation_report(
    tmp_data_dir: Path, fake_env: None
) -> None:
    result = CliRunner().invoke(cli, ["governance", "audit", "trend"])
    assert result.exit_code == 0, result.output
    assert "validation_report.json" in result.output
    assert "missing" in result.output.lower()


def test_strategies_shows_governance_column(tmp_data_dir: Path, fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["strategies"])
    assert result.exit_code == 0, result.output
    assert "Governance" in result.output
