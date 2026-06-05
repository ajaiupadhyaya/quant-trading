"""Claude decision-maker Phase A: structured brief, audit log, fail-open."""

from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant.analyst.advisor import AdvisorBrief, Proposals, advise, propose

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


# --- Phase B: advise-and-log proposals -------------------------------------

PROPOSAL_INPUT = {
    "risk_throttle": 0.8,
    "allocation_tilt": [
        {"slug": "defensive-etf-allocation", "delta": 0.05},
        {"slug": "pairs", "delta": 0.2},  # NOT live -> must be discarded by the clamp
    ],
    "should_halt": False,
    "halt_reason": "",
    "anomaly": "",
    "confidence": "medium",
    "rationale": "Mild caution given an unavailable regime read.",
}

LIVE = ["defensive-etf-allocation"]


def test_propose_clamps_tilts_to_live_strategies(tmp_path: Path) -> None:
    client = _FakeClient(name="submit_proposals", data=PROPOSAL_INPUT)
    p = propose(
        "f", "c", settings=_settings(), asof=ASOF, live_slugs=LIVE, client=client, data_dir=tmp_path
    )
    assert isinstance(p, Proposals)
    assert p.risk_throttle == 0.8
    # the non-live "pairs" tilt is discarded by governance; the live one survives
    assert p.allocation_tilt == {"defensive-etf-allocation": pytest.approx(0.05)}
    assert "pairs" in p.dropped_tilts
    # forced structured output
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_proposals"}


def test_propose_clamps_throttle_one_way(tmp_path: Path) -> None:
    bad = dict(PROPOSAL_INPUT, risk_throttle=2.5)  # never allowed to raise risk
    p = propose(
        "f",
        "c",
        settings=_settings(),
        asof=ASOF,
        live_slugs=LIVE,
        client=_FakeClient(name="submit_proposals", data=bad),
        data_dir=tmp_path,
    )
    assert p is not None and p.risk_throttle == 1.0


def test_propose_no_key_returns_none(tmp_path: Path) -> None:
    assert (
        propose(
            "f", "c", settings=_settings(key=None), asof=ASOF, live_slugs=LIVE, data_dir=tmp_path
        )
        is None
    )


def test_propose_logs_phase_b_applies_nothing(tmp_path: Path) -> None:
    propose(
        "f",
        "c",
        settings=_settings(),
        asof=ASOF,
        live_slugs=LIVE,
        client=_FakeClient(name="submit_proposals", data=PROPOSAL_INPUT),
        data_dir=tmp_path,
    )
    log = tmp_path / "analyst" / "decisions.jsonl"
    rec = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["phase"] == "B-advise-and-log"
    assert rec["applied"] is False
    assert rec["proposals"]["risk_throttle"] == 0.8
    assert rec["live_slugs"] == LIVE


def test_advise_builds_bounded_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SAFETY: a hung Claude call must never hold the shared 'batch' lock — the
    constructed client uses a short timeout + zero retries (not the SDK defaults)."""
    captured: dict[str, object] = {}

    class _Anthropic:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.messages = _FakeMessages("submit_brief", BRIEF_INPUT)

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    advise("f", "c", settings=_settings(), asof=ASOF, data_dir=tmp_path)
    assert captured.get("max_retries") == 0
    assert captured.get("timeout") == 20.0


def test_propose_uses_model_override(tmp_path: Path) -> None:
    """The intraday shadow log overrides the model with a cheaper one."""
    client = _FakeClient(name="submit_proposals", data=PROPOSAL_INPUT)
    propose(
        "f",
        "c",
        settings=_settings(),
        asof=ASOF,
        live_slugs=LIVE,
        client=client,
        data_dir=tmp_path,
        model="claude-haiku-4-5",
    )
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"
