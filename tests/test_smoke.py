"""End-to-end smoke: importing the package, running --help, listing strategies."""

from __future__ import annotations

import subprocess
import sys

import quant
from quant.cli import cli


def test_package_version() -> None:
    assert quant.__version__ == "0.1.0"


def test_cli_help_via_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "quant.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # exit code may be 0 (help shown) — what matters is that the module loads.
    assert "Usage" in result.stdout or "Usage" in result.stderr or result.returncode == 0


def test_every_subcommand_exists() -> None:
    expected = {
        "status",
        "backtest",
        "validate",
        "rebalance",
        "tearsheet",
        "journal",
        "monitor",
        "data",
        "strategies",
    }
    actual = set(cli.commands.keys())
    missing = expected - actual
    assert not missing, f"Missing subcommands: {missing}"
