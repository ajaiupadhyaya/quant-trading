"""Analyst digest: data gathering (pure), narration (mocked Claude), delivery."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant.analyst.digest import (
    DigestData,
    gather_digest_data,
    narrate,
    render_facts,
    run_digest,
)

ASOF = date(2026, 6, 2)


# --- fakes -----------------------------------------------------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        return _Resp(self._text)


class _FakeClient:
    def __init__(self, text: str = "Quiet paper day.") -> None:
        self.messages = _FakeMessages(text)


class _RaisingClient:
    class messages:  # noqa: N801
        @staticmethod
        def create(**kwargs: object) -> object:
            raise RuntimeError("api down")


class _FakeAlerts:
    def __init__(self) -> None:
        self.slack: list[tuple[str, object]] = []

    def send_slack(self, text: str, blocks: object = None) -> bool:
        self.slack.append((text, blocks))
        return True


def _settings(*, key: str | None = "sk-test", model: str = "claude-opus-4-8") -> SimpleNamespace:
    return SimpleNamespace(anthropic_api_key=key, anthropic_model=model)


# --- gather ----------------------------------------------------------------


def test_gather_prefers_injected_account_and_positions(tmp_path: Path) -> None:
    d = gather_digest_data(
        tmp_path,
        ASOF,
        dry_run=True,
        account={"equity": 1_010_000.0, "last_equity": 1_000_000.0, "cash": 500.0},
        live_positions=[("GLD", 800), ("DBC", 11000)],
        governance_live=["defensive-etf-allocation"],
    )
    assert d.equity == 1_010_000.0
    assert d.day_pl == 10_000.0
    assert abs(d.day_pl_pct - 0.01) < 1e-9
    assert d.positions == [("DBC", 11000), ("GLD", 800)]  # sorted
    assert d.governance_live == ["defensive-etf-allocation"]
    assert d.dry_run is True


def test_gather_reads_trades_jobs_and_guard_from_disk(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp(ASOF),
                "strategy": "defensive-etf-allocation",
                "symbol": "GLD",
                "side": "buy",
                "qty": 10,
                "client_order_id": "c1",
                "dry_run": True,
            }
        ]
    ).to_parquet(live / "trades.parquet")

    ops = tmp_path / "ops"
    (ops / "scheduler").mkdir(parents=True)
    (ops / "scheduler" / f"premarket-health.{ASOF.isoformat()}.json").write_text(
        json.dumps({"job": "premarket-health", "kind": "CATCH_UP", "exit_code": 0})
    )
    (ops / "monitor_status.json").write_text(
        json.dumps(
            {
                "worst_severity": "ok",
                "heartbeat": "hb",
                "halt_active": False,
                "outcomes": [{"name": "drift", "severity": "ok"}],
            }
        )
    )

    d = gather_digest_data(tmp_path, ASOF)
    assert len(d.orders) == 1 and d.orders[0]["symbol"] == "GLD" and d.orders[0]["dry_run"] is True
    assert d.jobs == [("premarket-health", "CATCH_UP", 0)]
    assert d.guard_worst_severity == "ok"
    assert d.guard_outcomes == [("drift", "ok")]


def test_gather_empty_dir_is_safe(tmp_path: Path) -> None:
    d = gather_digest_data(tmp_path, ASOF)
    assert d.equity is None and d.positions == [] and d.orders == [] and d.jobs == []
    assert d.day_pl is None


# --- render ----------------------------------------------------------------


def _data() -> DigestData:
    return DigestData(
        asof=ASOF,
        dry_run=True,
        equity=1_000_000.0,
        prev_equity=1_000_000.0,
        cash=12.0,
        governance_live=["defensive-etf-allocation"],
        positions=[("GLD", 800)],
        orders=[{"strategy": "x", "symbol": "GLD", "side": "buy", "qty": 800, "dry_run": True}],
        guard_worst_severity="ok",
        guard_heartbeat="hb",
        guard_outcomes=[("drift", "ok")],
        halt_active=False,
        jobs=[("daily-rebalance", "FRESH", 0)],
    )


def test_render_facts_has_key_lines() -> None:
    facts = render_facts(_data())
    assert "2026-06-02" in facts and "DRY-RUN" in facts
    assert "equity $1,000,000.00" in facts
    assert "GLD 800" in facts
    assert "BUY 800 GLD" in facts
    assert "Halt: none" in facts


def test_render_facts_flat_and_no_orders() -> None:
    d = DigestData(
        asof=ASOF, dry_run=False, equity=None, prev_equity=None, cash=None,
        governance_live=[], positions=[], orders=[], guard_worst_severity=None,
        guard_heartbeat=None, guard_outcomes=[], halt_active=True,
    )
    facts = render_facts(d)
    assert "flat" in facts and "Orders today: none" in facts and "ACTIVE" in facts


# --- narrate ---------------------------------------------------------------


def test_narrate_returns_none_without_key() -> None:
    assert narrate("facts", settings=_settings(key=None)) is None


def test_narrate_uses_injected_client() -> None:
    client = _FakeClient("Calm day — flat book, all guardrails green.")
    out = narrate("facts here", settings=_settings(), client=client)
    assert out == "Calm day — flat book, all guardrails green."
    # model + system cache_control + facts as the user message
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][0]["content"] == "facts here"


def test_narrate_swallows_api_error() -> None:
    assert narrate("facts", settings=_settings(), client=_RaisingClient()) is None


# --- run_digest (orchestration) -------------------------------------------


def test_run_digest_delivers_and_writes_artifact(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    art = tmp_path / "docs" / "analyst"
    res = run_digest(
        data_dir=tmp_path,
        asof=ASOF,
        settings=_settings(),
        alerts=alerts,
        artifact_dir=art,
        client=_FakeClient("NARRATIVE"),
        account={"equity": 1_000_000.0, "last_equity": 1_000_000.0, "cash": 1.0},
    )
    assert res.used_llm is True and res.body == "NARRATIVE" and res.delivered is True
    assert alerts.slack and "NARRATIVE" in alerts.slack[0][0]
    assert res.artifact_path == art / "2026-06-02.md"
    assert res.artifact_path.exists() and "NARRATIVE" in res.artifact_path.read_text()


def test_run_digest_dry_run_does_not_post(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    res = run_digest(
        data_dir=tmp_path,
        asof=ASOF,
        settings=_settings(),
        alerts=alerts,
        artifact_dir=tmp_path / "docs" / "analyst",
        client=_FakeClient("X"),
        dry_run=True,
    )
    assert res.delivered is False and alerts.slack == []
    assert res.artifact_path is not None and res.artifact_path.exists()


def test_run_digest_template_fallback_without_llm(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    res = run_digest(
        data_dir=tmp_path,
        asof=ASOF,
        settings=_settings(key=None),  # no key, no client → deterministic facts
        alerts=alerts,
        artifact_dir=tmp_path / "docs" / "analyst",
    )
    assert res.used_llm is False and res.body == res.facts
    assert res.delivered is True  # still posts the template digest
