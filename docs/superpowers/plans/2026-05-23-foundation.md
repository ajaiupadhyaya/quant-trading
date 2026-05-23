# Plan 1 of 6 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the repo skeleton, tooling, data layer, Alpaca client, and CLI scaffold so that Plans 2–6 (backtest engine, validation harness, strategies, TUI, live execution) can be built on top.

**Architecture:** Python 3.12 + `uv` + `pyproject.toml`. Click + Rich CLI with every subcommand wired to a strategy registry. Data layer reads/writes Parquet through a per-symbol cache, with Alpaca as primary and yfinance as backup. Alpaca execution wrapped with per-strategy `client_order_id` attribution. No strategy logic, no backtest engine, no TUI yet — those land in subsequent plans.

**Tech Stack:** Python 3.12, `uv`, Click, Rich, loguru, pydantic-settings, alpaca-py, yfinance, fredapi, pandas, pyarrow, numpy, pytest, hypothesis, ruff, mypy, freezegun, responses.

**Roadmap (informational):**
- **Plan 1 (this one): Foundation** — repo, data layer, Alpaca client, CLI scaffold
- Plan 2: Backtest engine + walk-forward + tear-sheet pipeline
- Plan 3: Combinatorial purged CV + deflated Sharpe + bootstrap + regime stress
- Plan 4: Strategies 1–3 (refined ports from Quant Lab v1 with SOTA enhancements)
- Plan 5: Strategies 4–5 (net-new TSMOM + HRP)
- Plan 6: Textual TUI + Alpaca live paper execution + GitHub Actions

---

## Setup Prerequisites

Before executing any task in this plan, the engineer needs:

1. **Alpaca paper trading API keys** — sign up at https://alpaca.markets, switch to Paper Trading, generate keys.
2. **FRED API key** — free at https://fred.stlouisfed.org/docs/api/api_key.html.
3. **`uv` installed** — `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`.
4. **A local `.env` file** — created during Task 1; never committed.

The engineer should run all tasks from the repo root (`~/Documents/quant-trading`). All shell commands assume that working directory unless stated otherwise.

---

## File Structure

Files created or modified by this plan, with each file's single responsibility:

```
quant-trading/
├── pyproject.toml                  ← uv project config: deps, ruff/mypy/pytest config, [project.scripts]
├── .python-version                 ← pins 3.12 for uv
├── .env.example                    ← documented env var template (no secrets)
├── .gitignore                      ← additions for .venv, data/raw, .env, etc.
├── README.md                       ← MODIFY: add CLI commands + status badges
├── .github/workflows/
│   └── ci.yml                      ← ruff + mypy + pytest on every push
├── data/
│   ├── universe/.gitkeep           ← committed dir (sp500 + ETF snapshots go here)
│   ├── backtests/.gitkeep          ← committed (tear-sheets later)
│   ├── live/.gitkeep               ← committed (equity + trades audit trail later)
│   ├── raw/.gitkeep                ← gitignored contents (regenerable)
│   ├── features/.gitkeep
│   ├── fundamentals/.gitkeep       ← gitignored contents
│   └── macro/.gitkeep              ← gitignored contents
├── quant/
│   ├── __init__.py                 ← __version__ export
│   ├── cli.py                      ← Click group + every subcommand stub
│   ├── strategies/
│   │   ├── __init__.py             ← REGISTRY + register decorator + list_strategies
│   │   └── base.py                 ← Strategy ABC + StrategySpec dataclass
│   ├── data/
│   │   ├── __init__.py
│   │   ├── universe.py             ← S&P 500 fetch + ETF universe + caching
│   │   ├── bars.py                 ← Alpaca + yfinance + parquet cache
│   │   ├── fundamentals.py         ← yfinance.info stub (Plan 4 replaces)
│   │   └── macro.py                ← FRED fetch + cache
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── alpaca.py               ← AlpacaClient wrapping TradingClient + market data
│   │   ├── orders.py               ← OrderTemplate + make_client_order_id
│   │   └── reconciler.py           ← target vs current → list of OrderTemplates
│   └── util/
│       ├── __init__.py
│       ├── logging.py              ← loguru wrapper + configure_logging
│       └── config.py               ← pydantic-settings BaseSettings
└── tests/
    ├── conftest.py                 ← shared fixtures + markers
    ├── test_smoke.py               ← end-to-end smoke: imports, CLI --help
    ├── util/
    │   ├── test_logging.py
    │   └── test_config.py
    ├── data/
    │   ├── test_universe.py
    │   ├── test_bars.py
    │   ├── test_fundamentals.py
    │   └── test_macro.py
    ├── execution/
    │   ├── test_alpaca.py
    │   ├── test_orders.py
    │   └── test_reconciler.py
    ├── strategies/
    │   └── test_registry.py
    └── test_cli.py
```

---

## Task 1: Project Skeleton + Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.env.example`
- Modify: `.gitignore`
- Create: `quant/__init__.py`
- Create: `quant/util/__init__.py`
- Create: `quant/data/__init__.py`
- Create: `quant/execution/__init__.py`
- Create: `quant/strategies/__init__.py` (full content in Task 8)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/{universe,raw,backtests,live,features,fundamentals,macro}/.gitkeep`

- [ ] **Step 1: Pin Python version**

Create `.python-version`:

```
3.12
```

- [ ] **Step 2: Create `pyproject.toml`**

Create `pyproject.toml`:

```toml
[project]
name = "quant"
version = "0.1.0"
description = "Systematic trading project: 5 strategies, paper-traded on Alpaca via GitHub Actions"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [{ name = "ajaiupadhyaya" }]

dependencies = [
    "click>=8.1",
    "rich>=13.7",
    "loguru>=0.7",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "alpaca-py>=0.30",
    "yfinance>=0.2.40",
    "fredapi>=0.5",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "numpy>=1.26",
    "python-dateutil>=2.9",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "lxml>=5.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "pytest-mock>=3.12",
    "hypothesis>=6.98",
    "ruff>=0.3",
    "mypy>=1.9",
    "types-requests",
    "types-python-dateutil",
    "freezegun>=1.4",
    "responses>=0.25",
    "pandas-stubs",
]

[project.scripts]
quant = "quant.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["quant"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "B", "UP", "N", "SIM", "RUF"]
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
exclude = ["tests/.*", "build/.*"]

[[tool.mypy.overrides]]
module = ["yfinance.*", "fredapi.*", "alpaca.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=quant --cov-report=term-missing"
markers = [
    "network: requires network access (skip in CI default)",
    "alpaca: requires Alpaca paper keys (skip in CI default)",
    "slow: slow test (skip in fast CI)",
]
```

- [ ] **Step 3: Create `.env.example`**

Create `.env.example`:

```bash
# Copy to .env and fill in your real values. .env is gitignored.

# Alpaca paper trading (https://alpaca.markets)
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# FRED (https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY=...

# Optional
LOG_LEVEL=INFO
QUANT_DATA_DIR=./data
```

- [ ] **Step 4: Update `.gitignore`**

Read the existing `.gitignore`, then append (do not replace):

```gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
*.egg-info/
.eggs/
build/
dist/

# Test / lint caches
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# IDE / OS
.vscode/
.idea/
.DS_Store

# Local secrets
.env
.env.local

# Data — regenerable from APIs
data/raw/**
!data/raw/.gitkeep
data/fundamentals/**
!data/fundamentals/.gitkeep
data/macro/**
!data/macro/.gitkeep
data/features/**
!data/features/.gitkeep
```

- [ ] **Step 5: Create package skeleton**

Create `quant/__init__.py`:

```python
"""quant — systematic trading project."""

__version__ = "0.1.0"
```

Create empty package markers — each with a single docstring line so it's not literally empty:

`quant/util/__init__.py`:
```python
"""Cross-cutting utilities: logging, config."""
```

`quant/data/__init__.py`:
```python
"""Data layer: bars, fundamentals, macro, universe."""
```

`quant/execution/__init__.py`:
```python
"""Alpaca execution + order reconciliation."""
```

`quant/strategies/__init__.py`:
```python
"""Strategy registry. Concrete strategies land in Plan 4."""
```

`tests/__init__.py`:
```python
"""Test package."""
```

- [ ] **Step 6: Create data directory placeholders**

For each of `universe`, `raw`, `backtests`, `live`, `features`, `fundamentals`, `macro`:

```bash
mkdir -p data/universe data/raw data/backtests data/live data/features data/fundamentals data/macro
touch data/universe/.gitkeep data/raw/.gitkeep data/backtests/.gitkeep data/live/.gitkeep data/features/.gitkeep data/fundamentals/.gitkeep data/macro/.gitkeep
```

- [ ] **Step 7: Create `tests/conftest.py`**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


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
```

- [ ] **Step 8: Bootstrap the environment with `uv`**

Run:

```bash
uv venv
uv sync --all-extras
```

Expected: virtualenv created in `.venv/`, all dependencies installed, no errors.

- [ ] **Step 9: Verify the package imports**

Run:

```bash
uv run python -c "import quant; print(quant.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 10: Verify ruff, mypy, pytest are runnable**

Run all three in sequence:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --collect-only
```

Expected: ruff passes with no findings. Format check passes. pytest collects 0 tests (no test files yet).

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml .python-version .env.example .gitignore quant/ tests/__init__.py tests/conftest.py data/
git commit -m "feat(foundation): project skeleton, tooling, package layout"
```

---

## Task 2: Logging + Config Utilities

**Files:**
- Create: `quant/util/logging.py`
- Create: `quant/util/config.py`
- Test: `tests/util/__init__.py`
- Test: `tests/util/test_logging.py`
- Test: `tests/util/test_config.py`

- [ ] **Step 1: Create test package marker**

Create `tests/util/__init__.py`:

```python
"""Tests for quant.util.*"""
```

- [ ] **Step 2: Write failing config tests**

Create `tests/util/test_config.py`:

```python
"""Tests for quant.util.config.Settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant.util.config import Settings


def test_settings_reads_required_env(fake_env: None) -> None:
    settings = Settings()
    assert settings.alpaca_api_key == "PKTEST"
    assert settings.alpaca_secret_key == "SECRETTEST"
    assert settings.fred_api_key == "FREDTEST"


def test_settings_paper_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setenv("FRED_API_KEY", "z")
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    settings = Settings()
    assert settings.alpaca_paper is True


def test_settings_data_dir_resolves(fake_env: None, tmp_data_dir: Path) -> None:
    settings = Settings()
    assert settings.data_dir == tmp_data_dir
    assert settings.data_dir.is_dir()


def test_settings_paper_url_for_paper(fake_env: None) -> None:
    settings = Settings()
    assert "paper-api.alpaca.markets" in settings.alpaca_base_url
```

- [ ] **Step 3: Run config tests to verify they fail**

Run:

```bash
uv run pytest tests/util/test_config.py -v
```

Expected: 4 FAILED — `ModuleNotFoundError: No module named 'quant.util.config'`.

- [ ] **Step 4: Implement `quant/util/config.py`**

Create `quant/util/config.py`:

```python
"""Application configuration via environment + .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config. Read once at startup; never mutate."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    alpaca_api_key: str = Field(..., description="Alpaca paper API key")
    alpaca_secret_key: str = Field(..., description="Alpaca paper secret key")
    alpaca_paper: bool = Field(default=True, description="Use paper account")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca REST base URL",
    )

    fred_api_key: str = Field(..., description="FRED API key")

    log_level: str = Field(default="INFO", description="loguru level")
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root data directory",
        validation_alias="QUANT_DATA_DIR",
    )
```

- [ ] **Step 5: Run config tests to verify they pass**

Run:

```bash
uv run pytest tests/util/test_config.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Write failing logging tests**

Create `tests/util/test_logging.py`:

```python
"""Tests for quant.util.logging."""

from __future__ import annotations

import io
import logging

import pytest

from quant.util.logging import configure_logging, logger


def test_logger_emits_at_configured_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logger.info("hello-info")
    logger.debug("hidden-debug")
    captured = capsys.readouterr()
    assert "hello-info" in captured.err
    assert "hidden-debug" not in captured.err


def test_logger_respects_lower_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG")
    logger.debug("now-visible")
    captured = capsys.readouterr()
    assert "now-visible" in captured.err


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")  # second call must not raise
```

- [ ] **Step 7: Run logging tests to verify they fail**

Run:

```bash
uv run pytest tests/util/test_logging.py -v
```

Expected: 3 FAILED — `ModuleNotFoundError: No module named 'quant.util.logging'`.

- [ ] **Step 8: Implement `quant/util/logging.py`**

Create `quant/util/logging.py`:

```python
"""Logging setup. Single loguru sink, level controlled by configure_logging."""

from __future__ import annotations

import sys

from loguru import logger as _logger

logger = _logger


def configure_logging(level: str = "INFO") -> None:
    """Reset loguru sinks and add a single stderr sink at the requested level.

    Idempotent — safe to call repeatedly.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
```

- [ ] **Step 9: Run logging tests to verify they pass**

Run:

```bash
uv run pytest tests/util/test_logging.py -v
```

Expected: 3 PASSED.

- [ ] **Step 10: Type-check**

Run:

```bash
uv run mypy quant/util/
```

Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add quant/util/logging.py quant/util/config.py tests/util/
git commit -m "feat(util): logging wrapper and pydantic-settings config"
```

---

## Task 3: Universe Module

**Files:**
- Create: `quant/data/universe.py`
- Test: `tests/data/__init__.py`
- Test: `tests/data/test_universe.py`

- [ ] **Step 1: Create data test package marker**

Create `tests/data/__init__.py`:

```python
"""Tests for quant.data.*"""
```

- [ ] **Step 2: Write failing universe tests**

Create `tests/data/test_universe.py`:

```python
"""Tests for quant.data.universe."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant.data.universe import (
    ETF_UNIVERSE,
    etf_universe,
    load_sp500_snapshot,
    save_sp500_snapshot,
    sp500_constituents,
)


def test_etf_universe_is_fixed_eight() -> None:
    tickers = etf_universe()
    assert tickers == ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "EFA", "EEM"]
    assert ETF_UNIVERSE == tickers  # module-level constant matches


def test_save_and_load_sp500_round_trip(tmp_data_dir: Path) -> None:
    sample = ["AAPL", "MSFT", "GOOGL"]
    path = save_sp500_snapshot(sample, snapshot_date=date(2026, 5, 23), data_dir=tmp_data_dir)
    assert path.exists()
    assert path.parent == tmp_data_dir / "universe"

    loaded = load_sp500_snapshot(snapshot_date=date(2026, 5, 23), data_dir=tmp_data_dir)
    assert loaded == sample


def test_load_sp500_snapshot_missing_raises(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sp500_snapshot(snapshot_date=date(1999, 1, 1), data_dir=tmp_data_dir)


def test_sp500_constituents_from_wikipedia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the parser extracts tickers from a pd.read_html-style result."""
    fake_table = pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT", "BRK.B", "GOOGL"],
            "Security": ["Apple", "Microsoft", "Berkshire", "Alphabet"],
        }
    )

    def fake_read_html(url: str, *args, **kwargs) -> list[pd.DataFrame]:
        assert "wikipedia.org" in url
        return [fake_table]

    monkeypatch.setattr("quant.data.universe.pd.read_html", fake_read_html)
    tickers = sp500_constituents()
    # Wikipedia uses "BRK.B" but Alpaca / yfinance use "BRK-B"
    assert "AAPL" in tickers
    assert "BRK-B" in tickers
    assert "BRK.B" not in tickers
```

- [ ] **Step 3: Run universe tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_universe.py -v
```

Expected: 4 FAILED — `ModuleNotFoundError: No module named 'quant.data.universe'`.

- [ ] **Step 4: Implement `quant/data/universe.py`**

Create `quant/data/universe.py`:

```python
"""S&P 500 + ETF universe lookups with on-disk snapshots."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant.util.config import Settings
from quant.util.logging import logger

ETF_UNIVERSE: list[str] = ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "EFA", "EEM"]

_WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def etf_universe() -> list[str]:
    """Return the canonical 8-ETF universe used by trend-following + HRP strategies."""
    return list(ETF_UNIVERSE)


def sp500_constituents() -> list[str]:
    """Fetch the current S&P 500 ticker list from Wikipedia.

    Wikipedia uses dotted tickers (BRK.B); Alpaca + yfinance use dashed (BRK-B).
    We normalize to the dashed form so downstream calls work without symbol mapping.
    """
    logger.info("Fetching S&P 500 constituents from Wikipedia")
    tables = pd.read_html(_WIKIPEDIA_SP500_URL)
    symbols_series = tables[0]["Symbol"].astype(str)
    return [s.strip().replace(".", "-") for s in symbols_series]


def _snapshot_path(snapshot_date: date, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir
    return base / "universe" / f"sp500_{snapshot_date.isoformat()}.csv"


def save_sp500_snapshot(
    tickers: list[str],
    snapshot_date: date,
    data_dir: Path | None = None,
) -> Path:
    """Persist a ticker list to data/universe/sp500_YYYY-MM-DD.csv."""
    path = _snapshot_path(snapshot_date, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(tickers, name="symbol").to_csv(path, index=False)
    logger.info("Saved S&P 500 snapshot to {} ({} tickers)", path, len(tickers))
    return path


def load_sp500_snapshot(
    snapshot_date: date,
    data_dir: Path | None = None,
) -> list[str]:
    """Read a previously-saved snapshot. Raises FileNotFoundError if absent."""
    path = _snapshot_path(snapshot_date, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"No S&P 500 snapshot at {path}")
    df = pd.read_csv(path)
    return df["symbol"].astype(str).tolist()
```

- [ ] **Step 5: Run universe tests to verify they pass**

Run:

```bash
uv run pytest tests/data/test_universe.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Type-check**

Run:

```bash
uv run mypy quant/data/universe.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add quant/data/universe.py tests/data/__init__.py tests/data/test_universe.py
git commit -m "feat(data): universe module (S&P 500 + ETF lookups, snapshots)"
```

---

## Task 4: Bars Module

**Files:**
- Create: `quant/data/bars.py`
- Test: `tests/data/test_bars.py`

- [ ] **Step 1: Write failing bars tests**

Create `tests/data/test_bars.py`:

```python
"""Tests for quant.data.bars."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant.data.bars import BarRequest, get_bars, _cache_path, _read_cache, _write_cache


def _fake_alpaca_frame(symbol: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [1_000_000 + i for i in range(len(dates))],
        },
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="timestamp"),
    )


def test_cache_round_trip(tmp_data_dir: Path) -> None:
    df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    path = _cache_path("AAPL", tmp_data_dir)
    _write_cache(df, path)
    assert path.exists()
    loaded = _read_cache(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_get_bars_cache_hit(tmp_data_dir: Path, fake_env: None) -> None:
    df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    _write_cache(df, _cache_path("AAPL", tmp_data_dir))

    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    with patch("quant.data.bars._fetch_alpaca") as mock_alpaca:
        result = get_bars(req)
    mock_alpaca.assert_not_called()
    assert ("AAPL", "close") in result.columns
    assert len(result) == 2


def test_get_bars_cache_miss_calls_alpaca(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    with patch("quant.data.bars._fetch_alpaca", return_value={"AAPL": fake_df}) as mock_alpaca:
        result = get_bars(req)
    mock_alpaca.assert_called_once()
    assert ("AAPL", "close") in result.columns
    assert _cache_path("AAPL", tmp_data_dir).exists()


def test_get_bars_alpaca_failure_falls_back_to_yfinance(
    tmp_data_dir: Path, fake_env: None
) -> None:
    req = BarRequest(symbols=["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 3))
    fake_df = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    with (
        patch("quant.data.bars._fetch_alpaca", side_effect=RuntimeError("alpaca down")),
        patch("quant.data.bars._fetch_yfinance", return_value={"AAPL": fake_df}) as mock_yf,
    ):
        result = get_bars(req)
    mock_yf.assert_called_once()
    assert ("AAPL", "close") in result.columns


def test_get_bars_multi_symbol_result_shape(tmp_data_dir: Path, fake_env: None) -> None:
    req = BarRequest(
        symbols=["AAPL", "MSFT"], start=date(2024, 1, 2), end=date(2024, 1, 3)
    )
    fake_aapl = _fake_alpaca_frame("AAPL", [date(2024, 1, 2), date(2024, 1, 3)])
    fake_msft = _fake_alpaca_frame("MSFT", [date(2024, 1, 2), date(2024, 1, 3)])
    with patch(
        "quant.data.bars._fetch_alpaca",
        return_value={"AAPL": fake_aapl, "MSFT": fake_msft},
    ):
        result = get_bars(req)
    # Multi-index columns: (symbol, field)
    assert set(result.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert "close" in result.columns.get_level_values(1)
```

- [ ] **Step 2: Run bars tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_bars.py -v
```

Expected: 5 FAILED — `ModuleNotFoundError: No module named 'quant.data.bars'`.

- [ ] **Step 3: Implement `quant/data/bars.py`**

Create `quant/data/bars.py`:

```python
"""Daily bar fetcher: Alpaca primary, yfinance backup, parquet cache.

The cache layout is per-symbol parquet files at data/raw/<symbol>.parquet.
Each file holds the full history we've seen — append-only growth, no rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from quant.util.config import Settings
from quant.util.logging import logger

_BAR_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class BarRequest:
    """A request for daily bars over [start, end] inclusive."""

    symbols: list[str]
    start: date
    end: date
    timeframe: str = "1Day"


def _cache_path(symbol: str, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir
    return base / "raw" / f"{symbol}.parquet"


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index, name="timestamp")
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _merge_cache(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _fetch_alpaca(
    symbols: list[str], start: date, end: date, settings: Settings
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars from Alpaca for the given symbols."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
    )
    bars = client.get_stock_bars(req)
    raw = bars.df  # multi-index (symbol, timestamp)

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for sym in symbols:
        if sym not in raw.index.get_level_values(0):
            continue
        sym_df = raw.xs(sym, level=0).copy()
        sym_df.index = pd.DatetimeIndex(sym_df.index.date, name="timestamp")
        sym_df = sym_df[[c for c in _BAR_COLUMNS if c in sym_df.columns]]
        out[sym] = sym_df
    return out


def _fetch_yfinance(
    symbols: list[str], start: date, end: date
) -> dict[str, pd.DataFrame]:
    """Fallback fetcher using yfinance."""
    import yfinance as yf

    raw = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    if len(symbols) == 1:
        df = raw.copy()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})
        df.index = pd.DatetimeIndex(df.index.date, name="timestamp")
        out[symbols[0]] = df[[c for c in _BAR_COLUMNS if c in df.columns]]
        return out

    for sym in symbols:
        if sym not in raw.columns.get_level_values(0):
            continue
        df = raw[sym].copy()
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.DatetimeIndex(df.index.date, name="timestamp")
        out[sym] = df[[c for c in _BAR_COLUMNS if c in df.columns]]
    return out


def get_bars(req: BarRequest) -> pd.DataFrame:
    """Return a wide DataFrame indexed by date with (symbol, field) columns.

    Cache strategy: read existing parquet, identify the gap vs the request range,
    fetch only what's missing, merge, and write back.
    """
    settings = Settings()
    data_dir = settings.data_dir

    frames: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for symbol in req.symbols:
        path = _cache_path(symbol, data_dir)
        if not path.exists():
            to_fetch.append(symbol)
            continue
        cached = _read_cache(path)
        have_start = cached.index.min().date() if len(cached) else None
        have_end = cached.index.max().date() if len(cached) else None
        if have_start is None or have_start > req.start or (have_end is not None and have_end < req.end):
            to_fetch.append(symbol)
        frames[symbol] = cached

    if to_fetch:
        try:
            fetched = _fetch_alpaca(to_fetch, req.start, req.end, settings)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch with fallback
            logger.warning("Alpaca fetch failed ({}); falling back to yfinance", exc)
            fetched = _fetch_yfinance(to_fetch, req.start, req.end)

        for sym, df in fetched.items():
            path = _cache_path(sym, data_dir)
            if path.exists():
                merged = _merge_cache(_read_cache(path), df)
            else:
                merged = df
            _write_cache(merged, path)
            frames[sym] = merged

    # Slice each frame to the requested window and stack columns
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        mask = (df.index >= pd.Timestamp(req.start)) & (df.index <= pd.Timestamp(req.end))
        sliced[sym] = df.loc[mask]

    if not sliced:
        return pd.DataFrame()
    return pd.concat(sliced, axis=1)
```

- [ ] **Step 4: Run bars tests to verify they pass**

Run:

```bash
uv run pytest tests/data/test_bars.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Type-check**

Run:

```bash
uv run mypy quant/data/bars.py
```

Expected: no errors (broad `Exception` catch is intentional and not a type issue).

- [ ] **Step 6: Commit**

```bash
git add quant/data/bars.py tests/data/test_bars.py
git commit -m "feat(data): daily bars module (Alpaca primary, yfinance backup, parquet cache)"
```

---

## Task 5: Fundamentals Module (yfinance Stub)

**Files:**
- Create: `quant/data/fundamentals.py`
- Test: `tests/data/test_fundamentals.py`

This is intentionally minimal. Plan 4 replaces yfinance with SEC EDGAR PIT data when the multi-factor strategy needs B/M, gross profitability, etc.

- [ ] **Step 1: Write failing fundamentals tests**

Create `tests/data/test_fundamentals.py`:

```python
"""Tests for quant.data.fundamentals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant.data.fundamentals import (
    book_to_market,
    get_fundamentals,
    gross_profitability,
)


def test_get_fundamentals_returns_dict(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "marketCap": 3_000_000_000_000,
        "trailingPE": 28.4,
        "priceToBook": 45.0,
        "returnOnEquity": 1.5,
        "grossProfits": 175_000_000_000,
        "totalAssets": 350_000_000_000,
    }
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        info = get_fundamentals("AAPL")
    assert info["priceToBook"] == 45.0
    assert info["returnOnEquity"] == 1.5


def test_book_to_market_inverts_p_b(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {"priceToBook": 4.0}
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        assert book_to_market("AAPL") == pytest.approx(0.25)


def test_book_to_market_missing_returns_nan(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        result = book_to_market("AAPL")
    assert pd.isna(result)


def test_gross_profitability_divides_gp_by_assets(tmp_data_dir: Path, fake_env: None) -> None:
    mock_ticker = MagicMock()
    mock_ticker.info = {"grossProfits": 100.0, "totalAssets": 400.0}
    with patch("quant.data.fundamentals.yf.Ticker", return_value=mock_ticker):
        assert gross_profitability("AAPL") == pytest.approx(0.25)
```

- [ ] **Step 2: Run fundamentals tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_fundamentals.py -v
```

Expected: 4 FAILED — `ModuleNotFoundError: No module named 'quant.data.fundamentals'`.

- [ ] **Step 3: Implement `quant/data/fundamentals.py`**

Create `quant/data/fundamentals.py`:

```python
"""Fundamentals stub backed by yfinance.

NOTE: this is a Plan 1 (Foundation) stub. The multi-factor strategy in Plan 4
will replace yfinance with SEC EDGAR point-in-time data to avoid look-ahead
bias. Until then, this is fine for sanity-checking the rest of the plumbing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from quant.util.logging import logger


def get_fundamentals(symbol: str) -> dict[str, Any]:
    """Return yfinance's `info` dict for the symbol.

    Returns an empty dict on failure rather than raising — callers should
    handle missing fields with `dict.get(..., default)`.
    """
    try:
        return dict(yf.Ticker(symbol).info or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance fundamentals failed for {}: {}", symbol, exc)
        return {}


def book_to_market(symbol: str) -> float:
    """Book-to-market ratio, derived as 1 / priceToBook."""
    info = get_fundamentals(symbol)
    pb = info.get("priceToBook")
    if pb is None or pb == 0:
        return float("nan")
    return 1.0 / float(pb)


def gross_profitability(symbol: str) -> float:
    """Gross profitability = grossProfits / totalAssets (Novy-Marx 2013)."""
    info = get_fundamentals(symbol)
    gp = info.get("grossProfits")
    assets = info.get("totalAssets")
    if gp is None or assets is None or assets == 0:
        return float("nan")
    return float(gp) / float(assets)
```

- [ ] **Step 4: Run fundamentals tests to verify they pass**

Run:

```bash
uv run pytest tests/data/test_fundamentals.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add quant/data/fundamentals.py tests/data/test_fundamentals.py
git commit -m "feat(data): fundamentals stub backed by yfinance (Plan 4 replaces)"
```

---

## Task 6: Macro Module

**Files:**
- Create: `quant/data/macro.py`
- Test: `tests/data/test_macro.py`

- [ ] **Step 1: Write failing macro tests**

Create `tests/data/test_macro.py`:

```python
"""Tests for quant.data.macro."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant.data.macro import (
    FRED_SERIES,
    _cache_path,
    cpi,
    get_series,
    tenyear_yield,
    unemployment_rate,
    vix,
)


def _fake_series() -> pd.Series:
    return pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
        name="VIXCLS",
    )


def test_get_series_caches_to_parquet(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred):
        s = get_series("VIXCLS")
    assert _cache_path("VIXCLS", tmp_data_dir).exists()
    assert len(s) == 3


def test_get_series_uses_cache_on_second_call(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred) as fred_cls:
        get_series("VIXCLS")
        get_series("VIXCLS")  # should hit cache
    fred_cls.assert_called_once()


def test_vix_uses_vixcls_series_id(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred):
        vix()
    fred.get_series.assert_called_with(FRED_SERIES["vix"])


def test_helpers_dispatch_correct_series(tmp_data_dir: Path, fake_env: None) -> None:
    fred = MagicMock()
    fred.get_series.return_value = _fake_series()
    with patch("quant.data.macro.Fred", return_value=fred):
        tenyear_yield()
        unemployment_rate()
        cpi()
    called_ids = [call.args[0] for call in fred.get_series.call_args_list]
    assert FRED_SERIES["tenyear"] in called_ids
    assert FRED_SERIES["unemployment"] in called_ids
    assert FRED_SERIES["cpi"] in called_ids
```

- [ ] **Step 2: Run macro tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_macro.py -v
```

Expected: 4 FAILED — `ModuleNotFoundError: No module named 'quant.data.macro'`.

- [ ] **Step 3: Implement `quant/data/macro.py`**

Create `quant/data/macro.py`:

```python
"""FRED macro series fetcher with parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fredapi import Fred

from quant.util.config import Settings
from quant.util.logging import logger

FRED_SERIES: dict[str, str] = {
    "vix": "VIXCLS",
    "tenyear": "DGS10",
    "twoyear": "DGS2",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "fedfunds": "DFF",
    "gdp": "GDPC1",
}


def _cache_path(series_id: str, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else Settings().data_dir
    return base / "macro" / f"{series_id}.parquet"


def get_series(series_id: str) -> pd.Series:
    """Fetch a FRED series. Returns from cache if present, else fetches and caches."""
    settings = Settings()
    path = _cache_path(series_id, settings.data_dir)
    if path.exists():
        logger.debug("Macro cache hit: {}", series_id)
        return pd.read_parquet(path)[series_id]

    logger.info("Fetching FRED series {}", series_id)
    fred = Fred(api_key=settings.fred_api_key)
    series = fred.get_series(series_id)
    series.name = series_id

    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name=series_id).to_parquet(path)
    return series


def vix() -> pd.Series:
    """CBOE VIX (close)."""
    return get_series(FRED_SERIES["vix"])


def tenyear_yield() -> pd.Series:
    """10-year Treasury constant maturity rate."""
    return get_series(FRED_SERIES["tenyear"])


def unemployment_rate() -> pd.Series:
    """Civilian unemployment rate (UNRATE)."""
    return get_series(FRED_SERIES["unemployment"])


def cpi() -> pd.Series:
    """Consumer price index (CPIAUCSL, seasonally adjusted)."""
    return get_series(FRED_SERIES["cpi"])
```

- [ ] **Step 4: Run macro tests to verify they pass**

Run:

```bash
uv run pytest tests/data/test_macro.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add quant/data/macro.py tests/data/test_macro.py
git commit -m "feat(data): FRED macro series fetcher with parquet cache"
```

---

## Task 7: Alpaca Execution Client

**Files:**
- Create: `quant/execution/alpaca.py`
- Create: `quant/execution/orders.py`
- Create: `quant/execution/reconciler.py`
- Test: `tests/execution/__init__.py`
- Test: `tests/execution/test_alpaca.py`
- Test: `tests/execution/test_orders.py`
- Test: `tests/execution/test_reconciler.py`

- [ ] **Step 1: Create execution test package marker**

Create `tests/execution/__init__.py`:

```python
"""Tests for quant.execution.*"""
```

- [ ] **Step 2: Write failing order helpers tests**

Create `tests/execution/test_orders.py`:

```python
"""Tests for quant.execution.orders."""

from __future__ import annotations

import re
from datetime import date

from quant.execution.orders import OrderSide, OrderTemplate, make_client_order_id


def test_make_client_order_id_format() -> None:
    coid = make_client_order_id("momentum", "AAPL", date(2026, 5, 23))
    # <slug>-<YYYYMMDD>-<symbol>-<uuid8>
    assert re.match(r"^momentum-20260523-AAPL-[0-9a-f]{8}$", coid)


def test_make_client_order_id_is_unique_across_calls() -> None:
    a = make_client_order_id("momentum", "AAPL", date(2026, 5, 23))
    b = make_client_order_id("momentum", "AAPL", date(2026, 5, 23))
    assert a != b


def test_order_template_round_trip() -> None:
    tpl = OrderTemplate(
        symbol="AAPL",
        qty=10,
        side=OrderSide.BUY,
        strategy_slug="momentum",
    )
    assert tpl.symbol == "AAPL"
    assert tpl.qty == 10
    assert tpl.side is OrderSide.BUY
    assert tpl.strategy_slug == "momentum"
```

- [ ] **Step 3: Run order helpers tests to verify they fail**

Run:

```bash
uv run pytest tests/execution/test_orders.py -v
```

Expected: 3 FAILED — `ModuleNotFoundError`.

- [ ] **Step 4: Implement `quant/execution/orders.py`**

Create `quant/execution/orders.py`:

```python
"""Order-template dataclass + client_order_id helper for per-strategy attribution."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from datetime import date


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class OrderTemplate:
    """A target order to be submitted to Alpaca.

    `qty` is always a positive integer. `side` encodes direction.
    """

    symbol: str
    qty: int
    side: OrderSide
    strategy_slug: str


def make_client_order_id(strategy_slug: str, symbol: str, dt: date) -> str:
    """Format: <slug>-<YYYYMMDD>-<symbol>-<uuid8>.

    The slug prefix is how we attribute fills back to a specific strategy when
    multiple strategies share a single Alpaca account.
    """
    return f"{strategy_slug}-{dt:%Y%m%d}-{symbol}-{uuid.uuid4().hex[:8]}"
```

- [ ] **Step 5: Run order helpers tests to verify they pass**

Run:

```bash
uv run pytest tests/execution/test_orders.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Write failing reconciler tests**

Create `tests/execution/test_reconciler.py`:

```python
"""Tests for quant.execution.reconciler."""

from __future__ import annotations

from quant.execution.orders import OrderSide
from quant.execution.reconciler import reconcile


def test_no_change_emits_no_orders() -> None:
    orders = reconcile(
        target={"AAPL": 10}, current={"AAPL": 10}, strategy_slug="momentum"
    )
    assert orders == []


def test_buy_to_open_new_position() -> None:
    orders = reconcile(
        target={"AAPL": 10}, current={}, strategy_slug="momentum"
    )
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.BUY


def test_sell_to_close_position() -> None:
    orders = reconcile(
        target={}, current={"AAPL": 10}, strategy_slug="momentum"
    )
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.SELL


def test_resize_upward_emits_buy_delta() -> None:
    orders = reconcile(
        target={"AAPL": 15}, current={"AAPL": 10}, strategy_slug="momentum"
    )
    assert orders[0].qty == 5
    assert orders[0].side is OrderSide.BUY


def test_resize_downward_emits_sell_delta() -> None:
    orders = reconcile(
        target={"AAPL": 5}, current={"AAPL": 10}, strategy_slug="momentum"
    )
    assert orders[0].qty == 5
    assert orders[0].side is OrderSide.SELL


def test_flip_long_to_short_emits_two_orders() -> None:
    orders = reconcile(
        target={"AAPL": -5}, current={"AAPL": 10}, strategy_slug="momentum"
    )
    # First flatten 10 long, then open 5 short.
    assert len(orders) == 2
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.SELL
    assert orders[1].qty == 5
    assert orders[1].side is OrderSide.SELL


def test_flip_short_to_long_emits_two_orders() -> None:
    orders = reconcile(
        target={"AAPL": 5}, current={"AAPL": -10}, strategy_slug="momentum"
    )
    # First cover 10 short, then open 5 long.
    assert len(orders) == 2
    assert orders[0].qty == 10
    assert orders[0].side is OrderSide.BUY
    assert orders[1].qty == 5
    assert orders[1].side is OrderSide.BUY


def test_strategy_slug_propagates() -> None:
    orders = reconcile(
        target={"AAPL": 10}, current={}, strategy_slug="pairs"
    )
    assert orders[0].strategy_slug == "pairs"
```

- [ ] **Step 7: Run reconciler tests to verify they fail**

Run:

```bash
uv run pytest tests/execution/test_reconciler.py -v
```

Expected: 8 FAILED — `ModuleNotFoundError`.

- [ ] **Step 8: Implement `quant/execution/reconciler.py`**

Create `quant/execution/reconciler.py`:

```python
"""Compute the list of OrderTemplates needed to move from current to target positions."""

from __future__ import annotations

from quant.execution.orders import OrderSide, OrderTemplate


def reconcile(
    target: dict[str, int],
    current: dict[str, int],
    strategy_slug: str,
) -> list[OrderTemplate]:
    """Return the orders that transform `current` into `target`.

    Long-to-short or short-to-long flips are split into two orders (flatten, then
    reopen on the other side) so each fill is monotonically directional. Some
    brokers reject single orders that cross zero; this keeps us safe.
    """
    orders: list[OrderTemplate] = []
    symbols = sorted(set(target) | set(current))

    for sym in symbols:
        tgt = target.get(sym, 0)
        cur = current.get(sym, 0)
        if tgt == cur:
            continue

        # Crossing zero?
        if (cur > 0 and tgt < 0) or (cur < 0 and tgt > 0):
            # Step 1: flatten current
            flatten_side = OrderSide.SELL if cur > 0 else OrderSide.BUY
            orders.append(
                OrderTemplate(symbol=sym, qty=abs(cur), side=flatten_side, strategy_slug=strategy_slug)
            )
            # Step 2: open target on the other side
            open_side = OrderSide.BUY if tgt > 0 else OrderSide.SELL
            orders.append(
                OrderTemplate(symbol=sym, qty=abs(tgt), side=open_side, strategy_slug=strategy_slug)
            )
            continue

        delta = tgt - cur
        if delta > 0:
            # Need to increase long exposure (or reduce short)
            orders.append(
                OrderTemplate(symbol=sym, qty=abs(delta), side=OrderSide.BUY, strategy_slug=strategy_slug)
            )
        else:
            orders.append(
                OrderTemplate(symbol=sym, qty=abs(delta), side=OrderSide.SELL, strategy_slug=strategy_slug)
            )

    return orders
```

- [ ] **Step 9: Run reconciler tests to verify they pass**

Run:

```bash
uv run pytest tests/execution/test_reconciler.py -v
```

Expected: 8 PASSED.

- [ ] **Step 10: Write failing Alpaca client tests**

Create `tests/execution/test_alpaca.py`:

```python
"""Tests for quant.execution.alpaca."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant.execution.alpaca import AccountInfo, AlpacaClient, PositionRow
from quant.execution.orders import OrderSide, OrderTemplate


@pytest.fixture
def mock_trading_client() -> MagicMock:
    return MagicMock()


def test_account_returns_parsed_info(fake_env: None, mock_trading_client: MagicMock) -> None:
    mock_trading_client.get_account.return_value = MagicMock(
        equity="100000.00",
        last_equity="99500.00",
        buying_power="50000.00",
        pattern_day_trader=False,
        cash="25000.00",
        portfolio_value="100000.00",
    )
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        acct = client.account()
    assert isinstance(acct, AccountInfo)
    assert acct.equity == 100000.00
    assert acct.last_equity == 99500.00
    assert acct.pattern_day_trader is False


def test_positions_returns_list_of_position_rows(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    mock_trading_client.get_all_positions.return_value = [
        MagicMock(
            symbol="AAPL", qty="10", avg_entry_price="180.0",
            market_value="1850.00", unrealized_pl="50.00",
            side="long", current_price="185.0",
        ),
    ]
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        positions = client.positions()
    assert len(positions) == 1
    assert isinstance(positions[0], PositionRow)
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10
    assert positions[0].avg_entry_price == 180.0


def test_submit_order_includes_client_order_id_prefix(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        client.submit_order(
            OrderTemplate(symbol="AAPL", qty=10, side=OrderSide.BUY, strategy_slug="momentum")
        )
    # The submitted request should include a client_order_id prefixed with the strategy slug
    submitted = mock_trading_client.submit_order.call_args.args[0]
    assert submitted.client_order_id.startswith("momentum-")
    assert "AAPL" in submitted.client_order_id
    assert submitted.qty == 10


def test_submit_order_dry_run_does_not_call_api(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        client.submit_order(
            OrderTemplate(symbol="AAPL", qty=10, side=OrderSide.BUY, strategy_slug="momentum"),
            dry_run=True,
        )
    mock_trading_client.submit_order.assert_not_called()
```

- [ ] **Step 11: Run Alpaca client tests to verify they fail**

Run:

```bash
uv run pytest tests/execution/test_alpaca.py -v
```

Expected: 4 FAILED — `ModuleNotFoundError`.

- [ ] **Step 12: Implement `quant/execution/alpaca.py`**

Create `quant/execution/alpaca.py`:

```python
"""Alpaca client wrapper.

Thin layer over alpaca-py's TradingClient that:
- normalizes string-typed API responses into typed dataclasses,
- attaches client_order_id with per-strategy attribution,
- supports dry-run order submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from quant.execution.orders import OrderSide, OrderTemplate, make_client_order_id
from quant.util.config import Settings
from quant.util.logging import logger


@dataclass(frozen=True)
class AccountInfo:
    equity: float
    last_equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    pattern_day_trader: bool


@dataclass(frozen=True)
class PositionRow:
    symbol: str
    qty: int
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    current_price: float
    side: str  # "long" or "short"


def _f(x: object) -> float:
    return float(x) if x is not None else 0.0


def _i(x: object) -> int:
    return int(float(x)) if x is not None else 0


class AlpacaClient:
    """Wraps `alpaca-py` for the subset of operations we need."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._trading = TradingClient(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            paper=self.settings.alpaca_paper,
        )

    def account(self) -> AccountInfo:
        raw = self._trading.get_account()
        return AccountInfo(
            equity=_f(raw.equity),
            last_equity=_f(raw.last_equity),
            buying_power=_f(raw.buying_power),
            cash=_f(raw.cash),
            portfolio_value=_f(raw.portfolio_value),
            pattern_day_trader=bool(raw.pattern_day_trader),
        )

    def positions(self) -> list[PositionRow]:
        raw_positions = self._trading.get_all_positions()
        rows: list[PositionRow] = []
        for p in raw_positions:
            side = str(p.side).lower()
            qty = _i(p.qty)
            if side == "short":
                qty = -abs(qty)
            rows.append(
                PositionRow(
                    symbol=str(p.symbol),
                    qty=qty,
                    avg_entry_price=_f(p.avg_entry_price),
                    market_value=_f(p.market_value),
                    unrealized_pl=_f(p.unrealized_pl),
                    current_price=_f(p.current_price),
                    side=side,
                )
            )
        return rows

    def submit_order(self, order: OrderTemplate, *, dry_run: bool = False) -> str:
        """Submit a market order. Returns the client_order_id."""
        coid = make_client_order_id(order.strategy_slug, order.symbol, date.today())
        side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
        )
        if dry_run:
            logger.info("[DRY-RUN] would submit {} {} {} (coid={})", order.side, order.qty, order.symbol, coid)
            return coid
        self._trading.submit_order(req)
        logger.info("Submitted {} {} {} (coid={})", order.side, order.qty, order.symbol, coid)
        return coid
```

- [ ] **Step 13: Run Alpaca client tests to verify they pass**

Run:

```bash
uv run pytest tests/execution/test_alpaca.py -v
```

Expected: 4 PASSED.

- [ ] **Step 14: Type-check execution package**

Run:

```bash
uv run mypy quant/execution/
```

Expected: no errors.

- [ ] **Step 15: Commit**

```bash
git add quant/execution/ tests/execution/
git commit -m "feat(execution): Alpaca client + order templates + reconciler"
```

---

## Task 8: Strategy Registry + CLI Scaffold

**Files:**
- Create: `quant/strategies/base.py`
- Modify: `quant/strategies/__init__.py`
- Create: `quant/cli.py`
- Test: `tests/strategies/__init__.py`
- Test: `tests/strategies/test_registry.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Create strategies test package marker**

Create `tests/strategies/__init__.py`:

```python
"""Tests for quant.strategies.*"""
```

- [ ] **Step 2: Write failing strategy base + registry tests**

Create `tests/strategies/test_registry.py`:

```python
"""Tests for the strategy base + registry."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.strategies import REGISTRY, list_strategies, register
from quant.strategies.base import Strategy, StrategySpec


@register
class _ToyStrategy(Strategy):
    spec = StrategySpec(
        slug="toy",
        name="Toy Strategy (test only)",
        description="A placeholder used in tests.",
        universe=["AAPL", "MSFT"],
        rebalance_frequency="daily",
    )

    def generate_signals(self, asof: date) -> pd.Series:
        return pd.Series({"AAPL": 1.0, "MSFT": -1.0})

    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        return {"AAPL": 10, "MSFT": -10}


def test_registry_contains_decorated_class() -> None:
    assert "toy" in REGISTRY
    assert REGISTRY["toy"] is _ToyStrategy


def test_list_strategies_returns_specs() -> None:
    specs = list_strategies()
    slugs = [s.slug for s in specs]
    assert "toy" in slugs


def test_strategy_can_be_instantiated_and_called() -> None:
    s = _ToyStrategy()
    signals = s.generate_signals(date(2026, 5, 23))
    assert signals.loc["AAPL"] == 1.0
    targets = s.target_positions(date(2026, 5, 23), equity=100_000)
    assert targets == {"AAPL": 10, "MSFT": -10}


def test_register_rejects_duplicate_slug() -> None:
    class _Dup(Strategy):
        spec = StrategySpec(
            slug="toy",  # duplicate
            name="dup",
            description="",
            universe=[],
            rebalance_frequency="daily",
        )

        def generate_signals(self, asof: date) -> pd.Series:
            return pd.Series()

        def target_positions(self, asof: date, equity: float) -> dict[str, int]:
            return {}

    with pytest.raises(ValueError, match="already registered"):
        register(_Dup)
```

- [ ] **Step 3: Run strategy tests to verify they fail**

Run:

```bash
uv run pytest tests/strategies/test_registry.py -v
```

Expected: failures because `quant.strategies.base` and the registry helpers don't exist yet.

- [ ] **Step 4: Implement `quant/strategies/base.py`**

Create `quant/strategies/base.py`:

```python
"""Strategy ABC + StrategySpec dataclass.

Concrete strategies land in Plans 4 and 5. Foundation only needs the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """Static metadata about a strategy."""

    slug: str
    name: str
    description: str
    universe: list[str]
    rebalance_frequency: str  # "daily" | "weekly" | "monthly"
    enabled_live: bool = field(default=False)


class Strategy(ABC):
    """Base class for all strategies. Concrete strategies subclass and register."""

    spec: StrategySpec  # class attribute provided by subclass

    @abstractmethod
    def generate_signals(self, asof: date) -> pd.Series:
        """Return a Series indexed by symbol with the signal score for each name."""

    @abstractmethod
    def target_positions(self, asof: date, equity: float) -> dict[str, int]:
        """Return target whole-share positions keyed by symbol.

        Positive = long, negative = short, missing/zero = no position.
        """
```

- [ ] **Step 5: Implement `quant/strategies/__init__.py` (full content)**

Replace the contents of `quant/strategies/__init__.py` with:

```python
"""Strategy registry. Subclasses register themselves via @register."""

from __future__ import annotations

from quant.strategies.base import Strategy, StrategySpec

REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a Strategy subclass to the registry."""
    slug = cls.spec.slug
    if slug in REGISTRY:
        raise ValueError(f"Strategy slug {slug!r} is already registered")
    REGISTRY[slug] = cls
    return cls


def list_strategies() -> list[StrategySpec]:
    """Return the StrategySpecs for all registered strategies, sorted by slug."""
    return [REGISTRY[k].spec for k in sorted(REGISTRY)]


__all__ = ["REGISTRY", "Strategy", "StrategySpec", "list_strategies", "register"]
```

- [ ] **Step 6: Run strategy tests to verify they pass**

Run:

```bash
uv run pytest tests/strategies/test_registry.py -v
```

Expected: 4 PASSED.

- [ ] **Step 7: Write failing CLI tests**

Create `tests/test_cli.py`:

```python
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
    assert "unknown strategy" in result.output.lower() or "definitely-not-a-strategy" in result.output


def test_cli_status_renders(fake_env: None) -> None:
    mock_alpaca = MagicMock()
    mock_alpaca.account.return_value = MagicMock(
        equity=100000.0, last_equity=99500.0, buying_power=50000.0,
        cash=25000.0, portfolio_value=100000.0, pattern_day_trader=False,
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
```

- [ ] **Step 8: Run CLI tests to verify they fail**

Run:

```bash
uv run pytest tests/test_cli.py -v
```

Expected: failures — `ModuleNotFoundError: No module named 'quant.cli'`.

- [ ] **Step 9: Implement `quant/cli.py`**

Create `quant/cli.py`:

```python
"""Click CLI: top-level group + every subcommand wired to the strategy registry.

Foundation phase: most subcommands are stubs that raise `click.ClickException`
with a clear "not yet implemented in Plan N" message. `status` and `data` are
fully functional; the rest are scaffolded so the command surface is stable.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from quant.execution.alpaca import AlpacaClient
from quant.strategies import REGISTRY, list_strategies
from quant.util.config import Settings
from quant.util.logging import configure_logging

console = Console()


def _require_strategy(slug: str) -> str:
    if slug not in REGISTRY:
        known = ", ".join(s.slug for s in list_strategies()) or "(none registered)"
        raise click.ClickException(f"Unknown strategy {slug!r}. Known: {known}")
    return slug


@click.group(help="Systematic trading: backtest, validate, rebalance, monitor.")
@click.option("--log-level", default=None, help="Override log level (DEBUG/INFO/WARNING).")
def cli(log_level: str | None) -> None:
    settings = Settings.model_construct() if not _can_load_settings() else Settings()
    level = log_level or getattr(settings, "log_level", "INFO")
    configure_logging(level)


def _can_load_settings() -> bool:
    try:
        Settings()
        return True
    except Exception:  # noqa: BLE001 - CLI help path must not require env
        return False


@cli.command(help="Show Alpaca account snapshot and per-strategy attribution.")
def status() -> None:
    client = AlpacaClient()
    acct = client.account()
    positions = client.positions()

    acct_table = Table(title="Account", show_header=True)
    acct_table.add_column("Field")
    acct_table.add_column("Value", justify="right")
    acct_table.add_row("Equity", f"${acct.equity:,.2f}")
    acct_table.add_row("Last Equity", f"${acct.last_equity:,.2f}")
    acct_table.add_row("Cash", f"${acct.cash:,.2f}")
    acct_table.add_row("Buying Power", f"${acct.buying_power:,.2f}")
    acct_table.add_row("Pattern Day Trader", str(acct.pattern_day_trader))
    console.print(acct_table)

    if positions:
        pos_table = Table(title=f"Positions ({len(positions)})", show_header=True)
        for col in ("Symbol", "Qty", "Avg", "Last", "Mkt Value", "Unrealized PnL"):
            pos_table.add_column(col, justify="right" if col != "Symbol" else "left")
        for p in positions:
            pos_table.add_row(
                p.symbol,
                str(p.qty),
                f"${p.avg_entry_price:,.2f}",
                f"${p.current_price:,.2f}",
                f"${p.market_value:,.2f}",
                f"${p.unrealized_pl:,.2f}",
            )
        console.print(pos_table)
    else:
        console.print("[dim]No open positions.[/dim]")


@cli.command(help="Run full walk-forward backtest for <strategy> and open tear-sheet.")
@click.argument("strategy")
@click.option("--quick", is_flag=True, help="Skip combinatorial CV + bootstrap.")
def backtest(strategy: str, quick: bool) -> None:
    _require_strategy(strategy)
    raise click.ClickException(
        f"backtest is not implemented in Foundation. "
        f"Plan 2 (engine) will fill this in. (strategy={strategy}, quick={quick})"
    )


@cli.command(help="Run the full validation battery (walk-forward + CPCV + DSR + ...).")
@click.argument("strategy")
def validate(strategy: str) -> None:
    _require_strategy(strategy)
    raise click.ClickException(
        f"validate is not implemented in Foundation. "
        f"Plan 3 (validation) will fill this in. (strategy={strategy})"
    )


@cli.command(help="Run today's live rebalance across all enabled strategies.")
@click.option("--dry-run", is_flag=True, help="Print orders only; do not submit.")
def rebalance(dry_run: bool) -> None:
    raise click.ClickException(
        "rebalance is not implemented in Foundation. Plan 6 will wire it up."
    )


@cli.command(help="Open the HTML tear-sheet for <strategy>.")
@click.argument("strategy")
def tearsheet(strategy: str) -> None:
    _require_strategy(strategy)
    raise click.ClickException("tearsheet is not implemented in Foundation. Plan 2 will fill this in.")


@cli.command(help="Print the structured trade journal.")
@click.option("--since", default=None, help="Filter trades since YYYY-MM-DD.")
def journal(since: str | None) -> None:
    raise click.ClickException("journal is not implemented in Foundation. Plan 6 will fill this in.")


@cli.command(help="Open the Textual TUI monitor.")
def monitor() -> None:
    raise click.ClickException("monitor is not implemented in Foundation. Plan 6 will fill this in.")


@cli.group(help="Data subcommands.")
def data() -> None:
    pass


@data.command("refresh", help="Refresh all bar caches and macro series.")
def data_refresh() -> None:
    raise click.ClickException("data refresh is not implemented in Foundation. Plan 2 will fill this in.")


@data.command("inventory", help="Show what's currently on disk under data/.")
def data_inventory() -> None:
    settings = Settings()
    base = Path(settings.data_dir)
    table = Table(title=f"Data inventory ({base})", show_header=True)
    table.add_column("Subdirectory")
    table.add_column("Files", justify="right")
    table.add_column("Size (MB)", justify="right")
    for sub in ("universe", "raw", "backtests", "live", "features", "fundamentals", "macro"):
        d = base / sub
        if not d.exists():
            table.add_row(sub, "0", "0.00")
            continue
        files = [f for f in d.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        table.add_row(sub, str(len(files)), f"{size_mb:.2f}")
    console.print(table)


@cli.command(help="List all registered strategies.")
def strategies() -> None:
    table = Table(title="Registered strategies", show_header=True)
    table.add_column("Slug")
    table.add_column("Name")
    table.add_column("Rebalance")
    table.add_column("Universe size", justify="right")
    table.add_column("Live", justify="center")
    for spec in list_strategies():
        table.add_row(
            spec.slug,
            spec.name,
            spec.rebalance_frequency,
            str(len(spec.universe)),
            "yes" if spec.enabled_live else "no",
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    cli()
```

- [ ] **Step 10: Run CLI tests to verify they pass**

Run:

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 5+ PASSED (one parameterized test runs 8 times, so total around 12 PASSED).

- [ ] **Step 11: Verify the installed CLI works**

Run:

```bash
uv run quant --help
uv run quant strategies
```

Expected: help text prints, `strategies` shows an empty table (no strategies registered yet — toy strategy lives in tests only).

- [ ] **Step 12: Type-check**

Run:

```bash
uv run mypy quant/
```

Expected: no errors.

- [ ] **Step 13: Commit**

```bash
git add quant/strategies/ quant/cli.py tests/strategies/ tests/test_cli.py
git commit -m "feat(cli): strategy registry + Click scaffold with all subcommand stubs"
```

---

## Task 9: CI Workflow + End-to-End Smoke

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Write a smoke test**

Create `tests/test_smoke.py`:

```python
"""End-to-end smoke: importing the package, running --help, listing strategies."""

from __future__ import annotations

import subprocess
import sys

from click.testing import CliRunner

import quant
from quant.cli import cli


def test_package_version() -> None:
    assert quant.__version__ == "0.1.0"


def test_cli_help_via_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "quant.cli", "--help"],
        capture_output=True, text=True, check=False,
    )
    # exit code may be 0 (help shown) — what matters is that the module loads.
    assert "Usage" in result.stdout or "Usage" in result.stderr or result.returncode == 0


def test_every_subcommand_exists() -> None:
    expected = {"status", "backtest", "validate", "rebalance", "tearsheet",
                "journal", "monitor", "data", "strategies"}
    actual = set(cli.commands.keys())
    missing = expected - actual
    assert not missing, f"Missing subcommands: {missing}"
```

- [ ] **Step 2: Run smoke tests to verify they pass**

Run:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: 3 PASSED.

- [ ] **Step 3: Create the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.12

      - name: Install deps
        run: uv sync --all-extras

      - name: Ruff lint
        run: uv run ruff check .

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: mypy
        run: uv run mypy quant/

      - name: pytest
        env:
          # Dummy values so Settings() can instantiate inside tests that need it
          ALPACA_API_KEY: ci-dummy
          ALPACA_SECRET_KEY: ci-dummy
          ALPACA_PAPER: "true"
          FRED_API_KEY: ci-dummy
        run: uv run pytest -q --cov=quant --cov-report=xml -m "not network and not alpaca and not slow"

      - name: Upload coverage
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
```

- [ ] **Step 4: Update README with CLI quick-start**

Read the existing `README.md`, then replace the `## CLI` block (lines around 35–44 in the current file) with a richer block that shows working commands and notes which are stubs:

Replace the original `## CLI` section with:

```markdown
## CLI

The `quant` command-group is installed when you run `uv sync --all-extras`. From the repo root:

```bash
uv run quant --help                  # top-level help
uv run quant strategies              # list registered strategies (empty until Plan 4)
uv run quant status                  # Alpaca account + open positions (needs .env)
uv run quant data inventory          # show what's on disk under data/

# Stubs landing in later plans:
uv run quant backtest <strategy>     # Plan 2 — backtest engine
uv run quant validate <strategy>     # Plan 3 — validation harness
uv run quant rebalance --dry-run     # Plan 6 — live execution
uv run quant tearsheet <strategy>    # Plan 2 — tear-sheet viewer
uv run quant journal                 # Plan 6 — trade log
uv run quant monitor                 # Plan 6 — Textual TUI
```

## Local setup

```bash
git clone <repo>
cd quant-trading
cp .env.example .env                 # fill in Alpaca paper + FRED keys
uv venv && uv sync --all-extras
uv run pytest                        # run the unit tests
```
```

(If the existing README has additional sections after `## License & disclaimer`, leave them alone.)

- [ ] **Step 5: Run the full test suite + lint locally to confirm CI parity**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy quant/
uv run pytest -q
```

Expected: all four succeed.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml tests/test_smoke.py README.md
git commit -m "ci: GitHub Actions (ruff + mypy + pytest) + README quick-start"
```

- [ ] **Step 7: Verify in CI**

Push the branch and confirm the `ci` workflow goes green. If anything fails, fix and re-commit a new commit (no force-push, no `--amend`).

```bash
git push -u origin <branch>
# Open the Actions tab on GitHub and watch the run
```

---

## Self-Review Notes (already applied)

- **Spec coverage:** This plan implements §3.1 (data inventory: Alpaca + yfinance + FRED + minimal fundamentals), §3.2 (storage layout — committed dirs initialized; gitignore handles regenerable data), §5.1 (CLI command surface — every subcommand from the spec is wired, status + data inventory + strategies functional), §6 (repo layout — every directory under `quant/` exists), §7.1 Week 1 (data layer + CLI + Alpaca client). The backtest engine (Week 2 / Plan 2), validation harness (Week 3 / Plan 3), strategy logic (Weeks 4–5 / Plans 4–5), TUI + GitHub Actions for daily rebalance (Week 6 / Plan 6) are explicitly deferred.
- **Placeholders:** none. Every code step has full source. Stub CLI commands raise a clear `ClickException` naming which plan implements them — that's a documented behavior, not a placeholder.
- **Type consistency:** `AlpacaClient.account() -> AccountInfo`, `AlpacaClient.positions() -> list[PositionRow]`, `make_client_order_id(strategy, symbol, dt)`, `reconcile(target, current, strategy_slug)` — names match across tasks and tests.
- **Deferred (per spec §9):** algorithmic regime detection (Plan 3 decides), crypto sleeve default (Plan 5 decides), backtest start date (Plan 2 decides), Git LFS (Plan 6 decides if data/ approaches GitHub's soft limit).

---

## Definition of Done (Foundation)

After Task 9 is committed and CI is green:

1. `uv run quant --help` prints the full subcommand list.
2. `uv run quant status` returns Alpaca account info when `.env` has paper keys.
3. `uv run quant data inventory` prints a table of `data/` subdirectories.
4. `uv run pytest` passes all tests at ≥80% coverage on `quant/data/`, `quant/execution/`, `quant/strategies/`.
5. `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy quant/` all clean.
6. CI workflow green on push.

Plan 2 (backtest engine) can start the moment this is true.
