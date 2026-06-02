"""Claude decision-maker Phase A: structured brief, audit log, fail-open."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from quant.analyst.advisor import AdvisorBrief, advise

ASOF = date(2026, 6, 2)

BRIEF_INPUT = {
    "headline": "Quiet day; defensive book steady.",
    "regime_read": "Calm regime, low VIX.",
    "risk_assessment": "No material concerns.",
    "suggested_risk_posture": 1.0,
    "confidence": "high",
    "watchlist": ["VIX", "SPY 200dma"],
    "rationale": "All governance gates green; one strategy live.",
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
    def __init__(self, name: str = "submit_brief", data: dict | None = None) -> None:
        self.messages = _FakeMessages(name, data if data is not None else BRIEF_INPUT)


class _RaisingClient:
    class messages:  # noqa: N801
        @staticmethod
        def create(**kwargs: object) -> object:
            raise RuntimeError("api down")


def _settings(key: str | None = "sk-test") -> SimpleNamespace:
    return SimpleNamespace(anthropic_api_key=key, anthropic_model="claude-opus-4-8")


# --- tests -----------------------------------------------------------------


def test_advise_parses_structured_brief_and_forces_tool(tmp_path: Path) -> None:
    client = _FakeClient()
    brief = advise(
        "facts", "context", settings=_settings(), asof=ASOF, client=client, data_dir=tmp_path
    )
    assert isinstance(brief, AdvisorBrief)
    assert brief.headline.startswith("Quiet day")
    assert brief.suggested_risk_posture == 1.0
    assert brief.watchlist == ["VIX", "SPY 200dma"]
    # structured output is forced via tool_choice
    kwargs = client.messages.calls[0]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_brief"}
    # render is Slack-friendly
    assert "*Quiet day" in brief.render()


def test_advise_clamps_posture_into_unit_interval(tmp_path: Path) -> None:
    bad = dict(BRIEF_INPUT, suggested_risk_posture=2.5)
    brief = advise(
        "f", "c", settings=_settings(), asof=ASOF, client=_FakeClient(data=bad), data_dir=tmp_path
    )
    assert brief is not None and brief.suggested_risk_posture == 1.0


def test_advise_no_key_returns_none(tmp_path: Path) -> None:
    assert advise("f", "c", settings=_settings(key=None), asof=ASOF, data_dir=tmp_path) is None


def test_advise_failopen_on_api_error_and_logs(tmp_path: Path) -> None:
    out = advise(
        "f", "c", settings=_settings(), asof=ASOF, client=_RaisingClient(), data_dir=tmp_path
    )
    assert out is None
    # the failure is recorded in the audit log (never silent)
    log = tmp_path / "analyst" / "decisions.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["applied"] is False
    assert rec["brief"] is None
    assert rec["error"] is not None


def test_advise_writes_audit_log_on_success(tmp_path: Path) -> None:
    advise("f", "c", settings=_settings(), asof=ASOF, client=_FakeClient(), data_dir=tmp_path)
    log = tmp_path / "analyst" / "decisions.jsonl"
    rec = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["applied"] is False  # Phase A applies nothing
    assert rec["brief"]["headline"].startswith("Quiet day")
    assert rec["model"] == "claude-opus-4-8"
    assert rec["phase"] == "A-advisory"
