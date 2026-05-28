from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from quant.backtest.engine import BacktestResult
from quant.cli import cli


def _fake_result(n: int = 300) -> BacktestResult:
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx, name="returns")
    equity = (1.0 + rets).cumprod() * 100_000.0
    from quant.backtest.engine import BacktestConfig

    return BacktestResult(
        equity_curve=equity,
        returns=rets,
        positions=pd.DataFrame(index=idx),
        trades=pd.DataFrame(),
        config=BacktestConfig(),
        starting_equity=100_000.0,
        ending_equity=float(equity.iloc[-1]),
    )


def test_sizing_compare_smoke(
    tmp_data_dir: Path, fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tmp_data_dir points QUANT_DATA_DIR at a sandbox + makes the dir tree;
    # fake_env supplies dummy Alpaca/FRED creds so Settings() doesn't fail.
    # Stub bars + backtest so the test doesn't hit the network.
    import quant.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "get_bars", lambda *a, **k: pd.DataFrame({("SPY", "close"): [1.0, 2.0]})
    )
    monkeypatch.setattr(cli_mod, "_run_single_backtest", lambda *a, **k: _fake_result())

    runner = CliRunner()
    end = date.today()
    start = end - timedelta(days=900)
    res = runner.invoke(
        cli,
        ["sizing", "compare", "trend", "--start", str(start), "--end", str(end)],
    )
    assert res.exit_code == 0, res.output
    assert "Sharpe" in res.output
    assert "Gross exposure" in res.output
    # registry record appended
    reg = tmp_data_dir / "research" / "experiments.jsonl"
    assert reg.exists()
