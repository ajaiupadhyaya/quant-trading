"""Tests for the live rebalance orchestrator.

These exercise the real strategy registry + reconciler but stub:
- AlpacaClient (no network)
- bar fetching (synthetic bars)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant.execution.alpaca import AccountInfo
from quant.execution.orders import OrderTemplate
from quant.live import bookkeeping, rebalance
from quant.live.bookkeeping import read_equity, read_trades
from quant.live.rebalance import run_rebalance
from quant.util.config import Settings
from tests.conftest import synthetic_bars


class _StubAlpacaClient:
    def __init__(self, equity: float = 100_000.0) -> None:
        self._equity = equity
        self.submitted: list[OrderTemplate] = []
        self.dry_run_flags: list[bool] = []

    def account(self) -> AccountInfo:
        return AccountInfo(
            equity=self._equity,
            last_equity=self._equity,
            buying_power=self._equity * 2,
            cash=self._equity * 0.1,
            portfolio_value=self._equity,
            pattern_day_trader=False,
        )

    def positions(self) -> list:  # type: ignore[type-arg]
        return []

    def submit_order(self, order: OrderTemplate, *, dry_run: bool = False) -> str:
        self.submitted.append(order)
        self.dry_run_flags.append(dry_run)
        return f"{order.strategy_slug}-stub-{order.symbol}"


@pytest.fixture
def fake_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "x")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("FRED_API_KEY", "x")
    monkeypatch.setenv("QUANT_DATA_DIR", str(data))
    return Settings()  # type: ignore[call-arg]


@pytest.fixture
def patched_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``_bars_for`` so we never hit the network during rebalance tests."""

    def _stub(strategy_cls: Any, asof: date, history_days: int) -> pd.DataFrame:
        return synthetic_bars(
            list(strategy_cls.spec.universe),
            date(asof.year - 3, asof.month, asof.day),
            asof,
            seed=21,
        )

    monkeypatch.setattr(rebalance, "_bars_for", _stub)


def test_dry_run_does_not_persist_strategy_positions(
    fake_settings: Settings, patched_bars: None
) -> None:
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=["momentum"],
    )
    assert report.dry_run is True
    assert report.enabled_strategies == ["momentum"]
    # equity row should have been written even on dry-run
    eq = read_equity(fake_settings.data_dir)
    assert len(eq) == 1
    # strategy_positions parquet should NOT exist after a dry-run
    assert not (fake_settings.data_dir / "live" / "strategy_positions.parquet").exists()


def test_planning_mode_does_not_write_bookkeeping(
    fake_settings: Settings, patched_bars: None
) -> None:
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=["momentum"],
        record_bookkeeping=False,
    )

    assert report.dry_run is True
    assert report.total_orders == len(client.submitted)
    assert not (fake_settings.data_dir / "live" / "equity.parquet").exists()
    assert not (fake_settings.data_dir / "live" / "trades.parquet").exists()
    assert not (fake_settings.data_dir / "live" / "strategy_positions.parquet").exists()
    momentum_outcome = next(o for o in report.outcomes if o.slug == "momentum")
    for order in momentum_outcome.orders:
        assert momentum_outcome.reference_prices[order.symbol] > 0


def test_live_run_persists_positions_and_trades(
    fake_settings: Settings, patched_bars: None
) -> None:
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=["momentum"],
    )
    assert all(flag is False for flag in client.dry_run_flags)
    # If the strategy emitted targets, we should see trades + a snapshot.
    momentum_outcome = next(o for o in report.outcomes if o.slug == "momentum")
    if momentum_outcome.target:
        trades = read_trades(fake_settings.data_dir)
        assert not trades.empty
        assert (trades["strategy"] == "momentum").all()
        assert (
            bookkeeping.last_strategy_positions(fake_settings.data_dir, "momentum")
            == momentum_outcome.target
        )


def test_second_run_reconciles_against_prior_snapshot(
    fake_settings: Settings, patched_bars: None
) -> None:
    """A second rebalance with the same target should produce zero new orders."""
    client = _StubAlpacaClient()
    first = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=["momentum"],
    )
    n_first = client.submitted.__len__()
    momentum_first = next(o for o in first.outcomes if o.slug == "momentum")
    if not momentum_first.target:
        pytest.skip("strategy emitted no targets; cannot exercise reconciliation")

    # Patch the strategy to deterministically return the same target.
    from quant.strategies import REGISTRY

    fixed_target = momentum_first.target

    class _FrozenStrategy:
        spec = REGISTRY["momentum"].spec

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def target_positions(self, *_args: Any, **_kwargs: Any) -> dict[str, int]:
            return fixed_target

        @classmethod
        def build(cls, *_args: Any, **_kwargs: Any) -> _FrozenStrategy:
            return cls()

    original = REGISTRY["momentum"]
    REGISTRY["momentum"] = _FrozenStrategy  # type: ignore[assignment]
    try:
        run_rebalance(
            asof=date(2024, 7, 31),
            dry_run=False,
            client=client,  # type: ignore[arg-type]
            settings=fake_settings,
            strategies=["momentum"],
        )
    finally:
        REGISTRY["momentum"] = original
    # Second run should add zero new orders because the snapshot equals the target.
    assert len(client.submitted) == n_first


def test_rebalance_loads_chosen_params_when_present(
    fake_settings: Settings, patched_bars: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If data/backtests/<slug>/chosen_params.json exists, its ``latest`` is fed to build()."""
    import json

    chosen_dir = fake_settings.data_dir / "backtests" / "momentum"
    chosen_dir.mkdir(parents=True, exist_ok=True)
    (chosen_dir / "chosen_params.json").write_text(
        json.dumps({"latest": {"top_pct": 0.99}, "windows": []})
    )

    captured: dict[str, Any] = {}
    from quant.strategies import REGISTRY

    original_build = REGISTRY["momentum"].build

    @classmethod  # type: ignore[misc]
    def _capturing_build(cls, bars, params=None):  # type: ignore[no-untyped-def]
        captured["params"] = params
        return original_build(bars=bars, params=params)

    monkeypatch.setattr(REGISTRY["momentum"], "build", _capturing_build)

    client = _StubAlpacaClient()
    run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=["momentum"],
    )
    assert captured.get("params") == {"top_pct": 0.99}


def test_no_enabled_strategies_is_noop(fake_settings: Settings, patched_bars: None) -> None:
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=[],
    )
    assert report.outcomes == []
    assert client.submitted == []


def _write_state_file(data_dir: Path, states: dict[str, str]) -> None:
    import json

    gov = data_dir / "governance"
    gov.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: {
                "slug": slug,
                "state": state,
                "evaluated_at": "2026-05-26T00:00:00",
                "validation_age_days": 1,
                "reason_codes": [] if state == "live" else ["failed_gate_regime"],
                "reason": "ok" if state == "live" else "blocked",
                "code_enabled_live": True,
                "manual_block": False,
            }
            for slug, state in states.items()
        },
    }
    (gov / "strategy_states.json").write_text(json.dumps(payload))


def test_default_rebalance_uses_only_governance_live_strategies(
    fake_settings: Settings, patched_bars: None
) -> None:
    _write_state_file(fake_settings.data_dir, {"momentum": "quarantined", "trend": "live"})
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == ["trend"]


def test_missing_governance_artifacts_fail_closed_for_default_rebalance(
    fake_settings: Settings, patched_bars: None
) -> None:
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == []
    assert report.skipped_reason is not None
    assert "governance" in report.skipped_reason.lower()


def test_include_quarantined_requires_dry_run(fake_settings: Settings, patched_bars: None) -> None:
    _write_state_file(fake_settings.data_dir, {"momentum": "quarantined"})
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        include_quarantined=True,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == []
    assert report.skipped_reason is not None
    assert "dry-run" in report.skipped_reason.lower()


def test_emergency_halt_blocks_non_dry_run(fake_settings: Settings, patched_bars: None) -> None:
    from quant.governance.halt import set_halt

    set_halt(fake_settings.data_dir, reason="operator stop")
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        strategies=["momentum"],
        skip_safety_checks=True,
    )

    assert report.enabled_strategies == []
    assert report.skipped_reason is not None
    assert "halt" in report.skipped_reason.lower()
    assert client.submitted == []


def test_dry_run_can_include_quarantined_for_observation(
    fake_settings: Settings, patched_bars: None
) -> None:
    _write_state_file(fake_settings.data_dir, {"momentum": "quarantined"})
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        include_quarantined=True,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == ["momentum"]
