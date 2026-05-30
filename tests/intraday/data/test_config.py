# tests/intraday/data/test_config.py
from datetime import date
from pathlib import Path

from quant.intraday.data.config import DEFAULT_UNIVERSE, IntradayConfig, partition_path


def test_default_universe_is_liquid_and_deduped():
    assert "SPY" in DEFAULT_UNIVERSE and "AAPL" in DEFAULT_UNIVERSE
    assert len(DEFAULT_UNIVERSE) == len(set(DEFAULT_UNIVERSE))
    assert 50 <= len(DEFAULT_UNIVERSE) <= 150


def test_partition_path_layout():
    root = Path("/data/intraday")
    p = partition_path(root, "trades", "AAPL", date(2023, 6, 1))
    assert p == root / "trades" / "symbol=AAPL" / "date=2023-06-01.parquet"


def test_config_defaults(tmp_path):
    cfg = IntradayConfig(data_root=tmp_path)
    assert cfg.hot_window_days == 5
    assert cfg.universe == DEFAULT_UNIVERSE
    assert cfg.data_root == tmp_path
