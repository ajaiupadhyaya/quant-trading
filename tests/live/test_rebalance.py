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

    def submit_order(
        self, order: OrderTemplate, *, asof: date | None = None, dry_run: bool = False
    ) -> str:
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


# ---------------------------------------------------------------------------
# Orphan wind-down integration tests
# ---------------------------------------------------------------------------


class _StubAlpacaClientWithPositions(_StubAlpacaClient):
    """Extended stub that returns specific Alpaca positions for reconciliation."""

    def __init__(self, equity: float = 100_000.0, alpaca_positions: list | None = None) -> None:
        super().__init__(equity=equity)
        self._alpaca_positions = alpaca_positions or []

    def positions(self) -> list:  # type: ignore[type-arg]
        return self._alpaca_positions


def test_orphan_winddown_exits_and_converges(fake_settings: Settings, patched_bars: None) -> None:
    """Live run: orphan (trend, QUARANTINED) with SPY:70 is reduced toward flat.

    Assertions:
    - report.winddown_outcomes contains an entry for "trend" with a SELL of SPY
    - After the run, last_strategy_positions(data_dir, "trend") snapshot has SPY=0
      (fully exited) or reduced (ADV-capped partial exit)
    - The stub recorded at least one submitted order attributed to "trend"
    """
    from quant.execution.alpaca import PositionRow
    from quant.live.bookkeeping import last_strategy_positions, write_strategy_positions

    asof = date(2024, 6, 28)

    # Governance: defensive-etf-allocation is LIVE, trend is QUARANTINED (orphan).
    _write_state_file(
        fake_settings.data_dir,
        {"defensive-etf-allocation": "live", "trend": "quarantined"},
    )

    # Seed a non-zero snapshot for trend so it is detected as an orphan.
    write_strategy_positions(fake_settings.data_dir, asof, "trend", {"SPY": 70})

    # The Alpaca aggregate must reflect those 70 shares so reconciliation passes.
    spy_position = PositionRow(
        symbol="SPY",
        qty=70,
        avg_entry_price=100.0,
        market_value=70 * 100.0,
        unrealized_pl=0.0,
        current_price=100.0,
        side="long",
    )

    client = _StubAlpacaClientWithPositions(
        equity=100_000.0,
        alpaca_positions=[spy_position],
    )

    report = run_rebalance(
        asof=asof,
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )

    # There must be at least one wind-down outcome for "trend".
    wd_slugs = [o.slug for o in report.winddown_outcomes]
    assert "trend" in wd_slugs, f"expected 'trend' in winddown_outcomes, got {wd_slugs}"

    trend_wd = next(o for o in report.winddown_outcomes if o.slug == "trend")
    assert trend_wd.error is None, f"wind-down errored: {trend_wd.error}"

    # Some SPY must have been exited (qty > 0 in exited dict).
    assert "SPY" in trend_wd.exited, f"expected SPY in exited, got {trend_wd.exited}"
    assert trend_wd.exited["SPY"] > 0

    # Under netting the submitted order's strategy_slug is the largest contributor.
    # When defensive-etf-allocation also touches SPY the attributed slug may vary.
    # What we must assert is:
    #   (a) exactly one net order for SPY was submitted
    #   (b) the wind-down snapshot was updated to trend_wd.remaining (intent).
    spy_orders = [o for o in client.submitted if o.symbol == "SPY"]
    assert spy_orders, "expected at least one submitted net order for SPY (orphan wind-down)"
    assert len(spy_orders) == 1, (
        f"netting must collapse to exactly one SPY order, got {len(spy_orders)}: {spy_orders}"
    )

    # The snapshot must have been updated (remaining, possibly 0).
    new_snap = last_strategy_positions(fake_settings.data_dir, "trend")
    # remaining["SPY"] must be <= 70 (we only exit, never open).
    assert new_snap.get("SPY", 0) <= 70
    # The snapshot reflects intent (result.remaining) not just succeeded exits.
    assert new_snap.get("SPY", 0) == trend_wd.remaining.get("SPY", 0)


def test_orphan_winddown_partial_failure_snapshot_reflects_intent(
    fake_settings: Settings, patched_bars: None
) -> None:
    """Under collect-then-net-then-submit the orphan snapshot records INTENT (result.remaining)
    before any submission attempt, not just the successfully-submitted subset.

    Two-symbol orphan (trend with SPY:70, IEF:30).  The stub raises on IEF but
    succeeds for SPY.  After the live run BOTH symbols must show 0 in the snapshot
    because winddown_orders produced exit orders for both and we commit the full
    result.remaining (the orphan's intent) regardless of which net submit succeeds.

    (Previously the snapshot advanced only for successful submits.  Under netting
    the per-strategy snapshot is the single source of intent; the fail-safe was
    only needed when submissions were interleaved with snapshot writes.)
    """
    from quant.execution.alpaca import PositionRow
    from quant.live.bookkeeping import last_strategy_positions, write_strategy_positions

    asof = date(2024, 6, 28)

    # Governance: defensive-etf-allocation is LIVE, trend is QUARANTINED (orphan).
    _write_state_file(
        fake_settings.data_dir,
        {"defensive-etf-allocation": "live", "trend": "quarantined"},
    )

    # Seed a two-symbol non-zero snapshot for trend.
    write_strategy_positions(fake_settings.data_dir, asof, "trend", {"SPY": 70, "IEF": 30})

    # Alpaca aggregate must reflect those shares so reconciliation passes.
    positions = [
        PositionRow(
            symbol="SPY",
            qty=70,
            avg_entry_price=100.0,
            market_value=70 * 100.0,
            unrealized_pl=0.0,
            current_price=100.0,
            side="long",
        ),
        PositionRow(
            symbol="IEF",
            qty=30,
            avg_entry_price=100.0,
            market_value=30 * 100.0,
            unrealized_pl=0.0,
            current_price=100.0,
            side="long",
        ),
    ]

    class _PartialFailClient(_StubAlpacaClientWithPositions):
        """Raises on submit_order for IEF; succeeds for everything else."""

        def submit_order(
            self, order: OrderTemplate, *, asof: date | None = None, dry_run: bool = False
        ) -> str:
            if order.symbol == "IEF":
                raise RuntimeError("simulated IEF submit failure")
            return super().submit_order(order, asof=asof, dry_run=dry_run)

    client = _PartialFailClient(equity=100_000.0, alpaca_positions=positions)

    report = run_rebalance(
        asof=asof,
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )

    # Wind-down outcome for "trend" must exist.
    wd_slugs = [o.slug for o in report.winddown_outcomes]
    assert "trend" in wd_slugs, f"expected 'trend' in winddown_outcomes, got {wd_slugs}"

    trend_wd = next(o for o in report.winddown_outcomes if o.slug == "trend")
    assert trend_wd.error is None, f"wind-down errored: {trend_wd.error}"

    # Under netting the orphan snapshot is the INTENT (result.remaining) and is
    # written before the net-submit loop.  Both SPY and IEF were fully exited by
    # winddown_orders, so remaining == 0 for both — regardless of whether the net
    # submit for IEF raised an exception.
    new_snap = last_strategy_positions(fake_settings.data_dir, "trend")
    assert new_snap.get("SPY", -1) == trend_wd.remaining.get("SPY", 0), (
        f"SPY snapshot must equal trend_wd.remaining (intent), got {new_snap.get('SPY')}"
    )
    assert new_snap.get("IEF", -1) == trend_wd.remaining.get("IEF", 0), (
        f"IEF snapshot must equal trend_wd.remaining (intent), got {new_snap.get('IEF')}"
    )
    # SPY submit succeeded so it must appear in the trade log.
    spy_submitted = [o for o in client.submitted if o.symbol == "SPY"]
    assert spy_submitted, "SPY net order must have been submitted"
    # IEF submit raised; it must NOT appear in the trade log.
    ief_submitted = [o for o in client.submitted if o.symbol == "IEF"]
    assert not ief_submitted, "IEF net order must NOT appear in submitted (submit raised)"


def test_orphan_winddown_dry_run_no_submit_no_zero(
    fake_settings: Settings, patched_bars: None
) -> None:
    """Dry run: orphan wind-down must NOT submit orders and must NOT update snapshot.

    The snapshot must remain exactly SPY:70 after the run.
    """
    from quant.execution.alpaca import PositionRow
    from quant.live.bookkeeping import last_strategy_positions, write_strategy_positions

    asof = date(2024, 6, 28)

    # Same governance setup as the live test.
    _write_state_file(
        fake_settings.data_dir,
        {"defensive-etf-allocation": "live", "trend": "quarantined"},
    )

    write_strategy_positions(fake_settings.data_dir, asof, "trend", {"SPY": 70})

    spy_position = PositionRow(
        symbol="SPY",
        qty=70,
        avg_entry_price=100.0,
        market_value=70 * 100.0,
        unrealized_pl=0.0,
        current_price=100.0,
        side="long",
    )
    client = _StubAlpacaClientWithPositions(
        equity=100_000.0,
        alpaca_positions=[spy_position],
    )

    report = run_rebalance(
        asof=asof,
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )

    # Wind-down outcome should still be recorded (the logic ran).
    wd_slugs = [o.slug for o in report.winddown_outcomes]
    assert "trend" in wd_slugs, f"expected 'trend' in winddown_outcomes, got {wd_slugs}"

    # In dry-run mode the stub records orders flagged dry_run=True, OR no order at
    # all — what matters is that the snapshot is NOT modified.
    # check that no real order was submitted (dry_run flag must be True for any trend order).
    trend_orders = [
        (o, dr)
        for o, dr in zip(client.submitted, client.dry_run_flags, strict=False)
        if o.strategy_slug == "trend"
    ]
    for _order, dr_flag in trend_orders:
        assert dr_flag is True, "dry-run wind-down order must have dry_run=True"

    # Snapshot must be unchanged — still SPY:70.
    snap = last_strategy_positions(fake_settings.data_dir, "trend")
    assert snap.get("SPY", 0) == 70, f"snapshot must remain SPY:70 in dry run, got {snap}"


# ---------------------------------------------------------------------------
# Netting integration test: opposing live-vs-orphan orders collapse to net
# ---------------------------------------------------------------------------


def test_netting_resolves_live_vs_orphan_conflict(
    fake_settings: Settings, patched_bars: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect-then-net-then-submit prevents opposing live+orphan orders for the same symbol.

    Setup:
    - defensive-etf-allocation (LIVE) target: {"DBC": 5}   → BUY DBC 5
    - trend (QUARANTINED orphan) snapshot:    {"DBC": 20, "VNQ": 8}
        → winddown produces SELL DBC 20, SELL VNQ 8

    Expected after netting:
    - DBC: BUY 5 - SELL 20 = SELL 15 → exactly ONE net order for DBC (side=sell, qty=15)
    - VNQ: no live order → SELL 8 passthrough → ONE order for VNQ (side=sell, qty=8)
    - Broker never sees an opposing BUY+SELL pair for DBC.
    """
    from quant.execution.alpaca import PositionRow
    from quant.live.bookkeeping import write_strategy_positions
    from quant.strategies import REGISTRY

    asof = date(2024, 6, 28)

    # Governance: defensive-etf-allocation LIVE, trend QUARANTINED (orphan).
    _write_state_file(
        fake_settings.data_dir,
        {"defensive-etf-allocation": "live", "trend": "quarantined"},
    )

    # Seed a two-symbol orphan snapshot for trend.
    write_strategy_positions(fake_settings.data_dir, asof, "trend", {"DBC": 20, "VNQ": 8})

    # Alpaca aggregate reflects the orphan holdings so reconciliation passes.
    positions = [
        PositionRow(
            symbol="DBC",
            qty=20,
            avg_entry_price=20.0,
            market_value=400.0,
            unrealized_pl=0.0,
            current_price=20.0,
            side="long",
        ),
        PositionRow(
            symbol="VNQ",
            qty=8,
            avg_entry_price=80.0,
            market_value=640.0,
            unrealized_pl=0.0,
            current_price=80.0,
            side="long",
        ),
    ]

    # Stub defensive-etf-allocation to deterministically target BUY DBC 5 only,
    # giving us a controlled opposing order for the netting assertion.
    original_cls = REGISTRY["defensive-etf-allocation"]

    class _FixedDefensive:
        spec = original_cls.spec

        def __init__(self, *_a: object, **_kw: object) -> None:
            pass

        def target_positions(self, *_a: object, **_kw: object) -> dict[str, int]:
            return {"DBC": 5}

        @classmethod
        def build(cls, *_a: object, **_kw: object) -> _FixedDefensive:
            return cls()

    monkeypatch.setitem(REGISTRY, "defensive-etf-allocation", _FixedDefensive)  # type: ignore[arg-type]

    client = _StubAlpacaClientWithPositions(
        equity=100_000.0,
        alpaca_positions=positions,
    )

    report = run_rebalance(
        asof=asof,
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )

    # Exactly one order per symbol must have been submitted (netting, not opposing pair).
    submitted_by_symbol: dict[str, list] = {}
    for o in client.submitted:
        submitted_by_symbol.setdefault(o.symbol, []).append(o)

    assert "DBC" in submitted_by_symbol, "DBC net order must have been submitted"
    dbc_orders = submitted_by_symbol["DBC"]
    assert len(dbc_orders) == 1, (
        f"expected exactly ONE net order for DBC, got {len(dbc_orders)}: {dbc_orders}"
    )
    net_dbc = dbc_orders[0]
    # Live BUY 5 - orphan SELL 20 = net SELL 15.
    assert str(net_dbc.side) == "sell", f"net DBC order must be SELL, got {net_dbc.side}"
    assert net_dbc.qty == 15, f"net DBC qty must be 15 (20 - 5), got {net_dbc.qty}"

    # VNQ: only the orphan holds it; no live order → passthrough SELL 8.
    # (winddown_orders may cap the qty by ADV; we assert <= 8 and side=sell.)
    assert "VNQ" in submitted_by_symbol, "VNQ passthrough sell must have been submitted"
    vnq_orders = submitted_by_symbol["VNQ"]
    assert len(vnq_orders) == 1, (
        f"expected exactly ONE net order for VNQ, got {len(vnq_orders)}: {vnq_orders}"
    )
    net_vnq = vnq_orders[0]
    assert str(net_vnq.side) == "sell", f"net VNQ order must be SELL, got {net_vnq.side}"
    assert 0 < net_vnq.qty <= 8, f"net VNQ qty must be in (0, 8], got {net_vnq.qty}"

    # The report must record the wind-down outcome for "trend".
    wd_slugs = [o.slug for o in report.winddown_outcomes]
    assert "trend" in wd_slugs, f"expected 'trend' in winddown_outcomes, got {wd_slugs}"
