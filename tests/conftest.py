"""Shared pytest fixtures and configuration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import Strategy, StrategySpec


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Provide an isolated data directory and point QUANT_DATA_DIR at it."""
    data = tmp_path / "data"
    for sub in ("universe", "raw", "backtests", "live", "features", "fundamentals", "macro"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANT_DATA_DIR", str(data))
    yield data


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate env with dummy credentials so Settings() doesn't fail."""
    monkeypatch.setenv("ALPACA_API_KEY", "PKTEST")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SECRETTEST")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("FRED_API_KEY", "FREDTEST")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def synthetic_bars(
    symbols: list[str],
    start: date,
    end: date,
    *,
    seed: int = 0,
    drift: float = 0.0003,
    vol: float = 0.01,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate deterministic wide-format daily bars for [start, end] business days.

    Returns a DataFrame indexed by date with MultiIndex columns (symbol, field)
    where field is in {open, high, low, close, volume}. Prices follow a geometric
    random walk; high/low are bracketed around close.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    if len(dates) == 0:
        return pd.DataFrame()
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        # Distinct seed per symbol so they're not perfectly correlated.
        shocks = rng.normal(loc=drift, scale=vol, size=len(dates))
        closes = start_price * np.exp(np.cumsum(shocks))
        opens = np.r_[closes[:1], closes[:-1]]  # open = prior close
        highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0, vol / 4, len(dates))))
        lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0, vol / 4, len(dates))))
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.full(len(dates), 1_000_000 + i, dtype=np.int64),
            },
            index=dates,
        )
        df.index.name = "timestamp"
        frames[sym] = df
    return pd.concat(frames, axis=1)


class EqualWeightStrategy(Strategy):
    """Test-only strategy: split equity equally across its universe at each rebalance."""

    spec: ClassVar[StrategySpec] = StrategySpec(
        slug="equal-weight-test",
        name="Equal Weight (test)",
        description="Test fixture: uniform allocation across the configured universe.",
        universe=["AAA", "BBB"],
        rebalance_frequency="monthly",
    )
    default_params: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        bars: pd.DataFrame,
        params: dict[str, object] | None = None,
        universe: list[str] | None = None,
    ) -> None:
        super().__init__(params=params)
        self._bars = bars
        self._universe = universe or list(self.spec.universe)

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({sym: 1.0 for sym in self._universe})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        ts = pd.Timestamp(asof)
        if ts not in self._bars.index:
            return {}
        per_name = equity / max(len(self._universe), 1)
        out: dict[str, int] = {}
        for sym in self._universe:
            price = float(self._bars[(sym, "close")].loc[ts])
            if price <= 0:
                continue
            out[sym] = int(per_name // price)
        return out


@pytest.fixture
def make_bars() -> Callable[..., pd.DataFrame]:
    """Factory fixture: tests call make_bars(symbols, start, end, seed=...) to get bars."""
    return synthetic_bars


@pytest.fixture
def equal_weight_strategy(
    make_bars: Callable[..., pd.DataFrame],
) -> tuple[EqualWeightStrategy, pd.DataFrame]:
    """A 2-symbol EqualWeight strategy + matching synthetic bars for a 1-year window."""
    bars = make_bars(["AAA", "BBB"], date(2024, 1, 1), date(2024, 12, 31), seed=42)
    strat = EqualWeightStrategy(bars=bars)
    return strat, bars
