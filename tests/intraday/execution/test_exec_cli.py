"""Tests for `quant intraday exec` CLI subgroup (Task 9)."""

from __future__ import annotations

from click.testing import CliRunner

from quant.intraday.cli import intraday


def test_exec_group_exists() -> None:
    r = CliRunner().invoke(intraday, ["exec", "--help"])
    assert r.exit_code == 0
    assert "frontier" in r.output and "schedule" in r.output


def test_schedule_prints_child_sizes() -> None:
    r = CliRunner().invoke(intraday, ["exec", "schedule", "--symbol", "QQQ",
                                      "--shares", "1000", "--horizon", "5"])
    assert r.exit_code == 0
    assert "QQQ" in r.output
    assert r.output.count("slice") >= 1 or "child" in r.output.lower()


def test_frontier_prints_points_and_baselines() -> None:
    r = CliRunner().invoke(intraday, ["exec", "frontier", "--symbol", "QQQ",
                                      "--shares", "1000"])
    assert r.exit_code == 0
    assert "frontier" in r.output.lower()
    assert "twap" in r.output.lower() and "immediate" in r.output.lower()
