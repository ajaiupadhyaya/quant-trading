"""`quant research signals` CLI: append/dry-run + exit-0 page-safety."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from quant.cli import cli
from quant.research import signals as sig
from quant.research.signals import build_market_signals, signals_path
from tests.research.conftest import close_panel, macro_series


def _computable_rec(asof: date | None = None):
    panel = close_panel(seed=5)
    idx = panel.index
    return build_market_signals(
        closes=panel,
        vix=macro_series(18.0, idx),
        dgs10=macro_series(4.0, idx),
        dgs2=macro_series(4.2, idx),
        asof=asof or idx[-1].date(),
    )


def test_dry_run_does_not_append(
    tmp_data_dir: Path, fake_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sig, "load_market_signals", lambda *a, **k: _computable_rec())
    res = CliRunner().invoke(cli, ["research", "signals", "--dry-run"])
    assert res.exit_code == 0
    assert not signals_path(tmp_data_dir).exists()


def test_appends_when_not_dry_run(
    tmp_data_dir: Path, fake_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sig, "load_market_signals", lambda *a, **k: _computable_rec())
    res = CliRunner().invoke(cli, ["research", "signals"])
    assert res.exit_code == 0
    lines = [ln for ln in signals_path(tmp_data_dir).read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert (tmp_data_dir / "research" / "signals_latest.json").exists()


def test_failopen_prints_unavailable_exit_zero(
    tmp_data_dir: Path, fake_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = build_market_signals(
        closes=pd.DataFrame(), vix=None, dgs10=None, dgs2=None, asof=date(2024, 6, 3)
    )
    monkeypatch.setattr(sig, "load_market_signals", lambda *a, **k: empty)
    res = CliRunner().invoke(cli, ["research", "signals"])
    assert res.exit_code == 0
    assert "unavailable" in res.output
    assert not signals_path(tmp_data_dir).exists()  # degraded -> no append


def test_exit_zero_even_when_loader_raises(
    tmp_data_dir: Path, fake_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(sig, "load_market_signals", _boom)
    res = CliRunner().invoke(cli, ["research", "signals"])
    assert res.exit_code == 0  # an unattended job must never page
    assert "unavailable" in res.output


def test_default_asof_is_today(
    tmp_data_dir: Path, fake_env: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, date] = {}

    def _capture(data_dir: object, d: date, **k: object):
        seen["asof"] = d
        return _computable_rec(d)

    monkeypatch.setattr(sig, "load_market_signals", _capture)
    CliRunner().invoke(cli, ["research", "signals", "--dry-run"])
    assert seen["asof"] == date.today()


def test_signals_show_errors_when_empty(tmp_data_dir: Path, fake_env: object) -> None:
    res = CliRunner().invoke(cli, ["research", "signals-show"])
    assert res.exit_code != 0  # human query command MAY raise (not in jobs.toml)
