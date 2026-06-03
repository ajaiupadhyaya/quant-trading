"""Intraday Claude watch: structured output, fail-open, bounded, non-spammy, read-only."""

from __future__ import annotations

import inspect
import json
import sys
import types
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant.analyst.digest import DigestData
from quant.analyst.watch import (
    WatchComment,
    comment,
    render_watch,
    run_watch,
)

ASOF = date(2026, 6, 3)

COMMENT_INPUT = {
    "headline": "Midday: defensive book steady.",
    "whats_moving": "Equity flat; commodities firm; all guardrails green.",
    "posture_note": "steady",
    "watchlist": ["VIX", "DBC"],
    "confidence": "medium",
}


# --- fakes -----------------------------------------------------------------


class _ToolBlock:
    def __init__(self, name: str, data: dict) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = data


class _Resp:
    def __init__(self, name: str, data: dict) -> None:
        self.content = [_ToolBlock(name, data)]


class _FakeMessages:
    def __init__(self, name: str, data: dict) -> None:
        self._name = name
        self._data = data
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> _Resp:
        self.calls.append(kwargs)
        return _Resp(self._name, self._data)


class _FakeClient:
    def __init__(self, name: str = "submit_commentary", data: dict | None = None) -> None:
        self.messages = _FakeMessages(name, data if data is not None else COMMENT_INPUT)


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


class _RaisingAlerts:
    def send_slack(self, text: str, blocks: object = None) -> bool:
        raise RuntimeError("slack down")


def _settings(key: str | None = "sk-test") -> SimpleNamespace:
    return SimpleNamespace(anthropic_api_key=key, anthropic_model="claude-opus-4-8")


def _decisions(data_dir: Path) -> list[dict]:
    path = data_dir / "analyst" / "decisions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_state(
    data_dir: Path,
    asof: date,
    *,
    posts_today: int = 0,
    last_post_at: datetime | None = None,
    last_hash: str | None = None,
) -> None:
    p = data_dir / "analyst" / "watch_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "session": asof.isoformat(),
                "posts_today": posts_today,
                "last_post_at": last_post_at.isoformat() if last_post_at else None,
                "last_hash": last_hash,
            }
        )
    )


# --- comment() (the Claude call) -------------------------------------------


def test_comment_parses_structured_note_and_forces_tool(tmp_path: Path) -> None:
    client = _FakeClient()
    cmt = comment(
        "facts", "ctx", settings=_settings(), asof=ASOF, slot="midday", client=client, data_dir=tmp_path
    )
    assert isinstance(cmt, WatchComment)
    assert cmt.headline.startswith("Midday")
    assert cmt.posture_note == "steady"
    assert cmt.watchlist == ["VIX", "DBC"]
    # structured output is forced — free text is never trusted
    kwargs = client.messages.calls[0]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_commentary"}
    assert "*[midday] Midday" in cmt.render("midday")


def test_comment_no_key_returns_none_and_makes_no_call(tmp_path: Path) -> None:
    assert comment("f", "c", settings=_settings(key=None), asof=ASOF, slot="open", data_dir=tmp_path) is None


def test_comment_failopen_on_api_error_and_logs(tmp_path: Path) -> None:
    cmt = comment(
        "f", "c", settings=_settings(), asof=ASOF, slot="open", client=_RaisingClient(), data_dir=tmp_path
    )
    assert cmt is None  # never raises
    rec = _decisions(tmp_path)[-1]
    assert rec["applied"] is False
    assert rec["phase"] == "watch-intraday"
    assert rec["comment"] is None
    assert rec["error"]


def test_comment_writes_audit_log_on_success(tmp_path: Path) -> None:
    comment("f", "c", settings=_settings(), asof=ASOF, slot="power-hour", client=_FakeClient(), data_dir=tmp_path)
    rec = _decisions(tmp_path)[-1]
    assert rec["applied"] is False
    assert rec["phase"] == "watch-intraday"
    assert rec["slot"] == "power-hour"
    assert rec["model"] == "claude-opus-4-8"  # no fast model set -> falls back to the default
    assert rec["comment"]["headline"].startswith("Midday")


def test_comment_uses_fast_model_when_set(tmp_path: Path) -> None:
    """Cost control: routine intraday calls use the cheaper 'fast' model when set,
    leaving Opus for the high-stakes brief."""
    client = _FakeClient()
    settings = SimpleNamespace(
        anthropic_api_key="sk-test",
        anthropic_model="claude-opus-4-8",
        anthropic_model_fast="claude-haiku-4-5",
    )
    comment("f", "c", settings=settings, asof=ASOF, slot="midday", client=client, data_dir=tmp_path)
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_comment_posture_note_is_sanitized(tmp_path: Path) -> None:
    bad = dict(COMMENT_INPUT, posture_note="SELL EVERYTHING")
    cmt = comment("f", "c", settings=_settings(), asof=ASOF, slot="m", client=_FakeClient(data=bad), data_dir=tmp_path)
    assert cmt is not None
    assert cmt.posture_note == "steady"  # off-enum value falls back to a safe word


def test_comment_builds_bounded_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SAFETY: a hung Claude call cannot hold the batch lock — the client MUST be
    constructed with a short timeout and zero retries (not the SDK defaults)."""
    captured: dict[str, object] = {}

    class _Anthropic:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.messages = _FakeMessages("submit_commentary", COMMENT_INPUT)

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    cmt = comment("f", "c", settings=_settings(), asof=ASOF, slot="midday", data_dir=tmp_path)
    assert isinstance(cmt, WatchComment)
    assert captured.get("max_retries") == 0
    assert captured.get("timeout") == 20.0


# --- render_watch() (deterministic fallback) -------------------------------


def test_render_watch_is_deterministic_fallback() -> None:
    d = DigestData(
        asof=ASOF, dry_run=False, equity=994_000.0, prev_equity=995_000.0, cash=100.0,
        governance_live=["defensive-etf-allocation"], positions=[("DBC", 11061)], orders=[],
        guard_worst_severity="ok", guard_heartbeat=None, guard_outcomes=[], halt_active=False,
    )
    text = render_watch(d, "midday")
    assert "[midday]" in text
    assert "Equity $994,000" in text
    assert "defensive-etf-allocation" in text
    assert "Guardrails: ok" in text


# --- run_watch() orchestration ---------------------------------------------


def test_run_watch_posts_when_new_and_writes_state(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(), alerts=alerts, slot="open",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC), client=_FakeClient(),
    )
    assert res.posted is True
    assert res.used_llm is True
    assert len(alerts.slack) == 1
    state = json.loads((tmp_path / "analyst" / "watch_state.json").read_text())
    assert state["posts_today"] == 1
    assert state["last_hash"]


def test_run_watch_respects_daily_cap_before_calling_claude(tmp_path: Path) -> None:
    _seed_state(tmp_path, ASOF, posts_today=4)
    client = _FakeClient()
    alerts = _FakeAlerts()
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(), alerts=alerts, slot="midday",
        now=datetime(2026, 6, 3, 18, 0, tzinfo=UTC), client=client, max_posts=4,
    )
    assert res.posted is False
    assert "cap" in (res.suppressed_reason or "")
    assert alerts.slack == []          # nothing posted
    assert client.messages.calls == []  # Claude never called — cost is bounded
    assert _decisions(tmp_path)[-1]["suppressed"]  # suppression is observable


def test_run_watch_respects_min_gap(tmp_path: Path) -> None:
    now = datetime(2026, 6, 3, 16, 0, tzinfo=UTC)
    _seed_state(tmp_path, ASOF, posts_today=1, last_post_at=now)
    client = _FakeClient()
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(), alerts=_FakeAlerts(), slot="midday",
        now=now, client=client, min_gap_min=30,
    )
    assert res.posted is False
    assert "min-gap" in (res.suppressed_reason or "")
    assert client.messages.calls == []


def test_run_watch_suppresses_duplicate_content(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    common = dict(data_dir=tmp_path, asof=ASOF, settings=_settings(), alerts=alerts, slot="midday")
    first = run_watch(**common, now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC), client=_FakeClient())
    # second run, past the min-gap, but the model returns identical commentary
    second = run_watch(**common, now=datetime(2026, 6, 3, 15, 0, tzinfo=UTC), client=_FakeClient())
    assert first.posted is True
    assert second.posted is False
    assert second.suppressed_reason == "duplicate content"
    assert len(alerts.slack) == 1  # only the first reached Slack


def test_run_watch_dry_run_does_not_post(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(), alerts=alerts, slot="open",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC), client=_FakeClient(), dry_run=True,
    )
    assert res.posted is False
    assert alerts.slack == []
    # still logged (the Claude call happened and is on the audit trail)
    assert any(r.get("phase") == "watch-intraday" and r.get("comment") for r in _decisions(tmp_path))


def test_run_watch_template_fallback_without_llm(tmp_path: Path) -> None:
    alerts = _FakeAlerts()
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(key=None), alerts=alerts, slot="midday",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
    )
    assert res.used_llm is False
    assert res.posted is True  # deterministic template still posts
    assert len(alerts.slack) == 1


def test_run_watch_never_raises_on_alerts_failure(tmp_path: Path) -> None:
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(), alerts=_RaisingAlerts(), slot="open",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC), client=_FakeClient(),
    )
    assert res.posted is False  # fail-open: a Slack hiccup can never crash a tick


def test_run_watch_failopen_on_empty_dir(tmp_path: Path) -> None:
    # no key, no client, empty data dir — must produce a result and never raise
    res = run_watch(
        data_dir=tmp_path, asof=ASOF, settings=_settings(key=None), alerts=_FakeAlerts(), slot="open",
        now=datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
    )
    assert res.body is not None


# --- safety: strictly read-only --------------------------------------------


def test_watch_module_imports_no_order_or_governance_write_paths() -> None:
    src = inspect.getsource(sys.modules["quant.analyst.watch"])
    forbidden = [
        "submit_order",
        "place_order",
        "run_rebalance",
        "quant.execution",
        "quant.live.rebalance",
        "write_strategy_states",
        "write_allocation",
        "set_halt",
        "save_halt",
        "build_governance_artifacts",
        "governance refresh",
    ]
    for sym in forbidden:
        assert sym not in src, f"watch.py must never reference an order/governance-write path: {sym!r}"


def test_watch_subcommand_registered() -> None:
    from quant.cli import analyst

    assert "watch" in analyst.commands
