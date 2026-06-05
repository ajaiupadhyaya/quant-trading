"""Phase A of the Claude decision-maker: a READ-ONLY, context-aware analyst brief.

Given the day's digest facts plus the richer :class:`AnalystContext`, Claude
returns a STRUCTURED brief (a schema-validated tool call — never trusted free
text) with a headline, a regime read, a risk assessment, a watchlist, and an
*advisory* risk posture in [0, 1] (1.0 = full size, <1.0 = a suggested de-risk
multiplier).

Hard safety boundary for Phase A: this module APPLIES NOTHING. It places no
orders, sets no halt, and changes no allocation. ``suggested_risk_posture`` is
recorded and shown to the operator only — nothing in the live path reads it.
Every call is appended to an immutable audit log so each decision is reviewable
and reproducible. Fail-open: any error / missing key returns ``None`` and the
caller falls back to the deterministic digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant.util.logging import logger

_MAX_TOKENS = 1500

_SYSTEM_PROMPT = """\
You are the senior analyst for a personal systematic PAPER-trading system. Each \
day you receive (1) a factual summary of the trading day and (2) a richer context \
pack: the current market regime + posteriors, per-strategy validation evidence and \
governance state, capital allocation, a macro snapshot, and recent execution \
quality. You connect these into one concise, decision-useful brief for the operator.

You are ADVISORY ONLY. You never place orders, never halt, never change allocation. \
Your `suggested_risk_posture` is a recommendation the operator reviews; it is NOT \
applied automatically. You may only ever suggest reducing risk (posture < 1.0) when \
the evidence/regime warrants caution — be conservative, not aggressive.

Rules:
- Ground every claim in the provided facts/context. Never invent numbers or strategies.
- Respect governance: only `defensive-etf-allocation` (or whatever is shown LIVE) is \
authorized; quarantined strategies are not eligible no matter how you read them.
- This is PAPER trading and may be in DRY-RUN; say so plainly and never imply real money.
- Be precise and brief. Submit your answer ONLY by calling the `submit_brief` tool."""

_BRIEF_TOOL: dict[str, Any] = {
    "name": "submit_brief",
    "description": "Submit the structured daily analyst brief. This is the only allowed output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": "One-sentence bottom line for the operator's phone.",
            },
            "regime_read": {
                "type": "string",
                "description": "Short interpretation of the current regime + macro.",
            },
            "risk_assessment": {
                "type": "string",
                "description": "What, if anything, warrants caution today.",
            },
            "suggested_risk_posture": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "ADVISORY ONLY, never applied automatically. 1.0 = full size; "
                    "<1.0 = a recommended de-risk multiplier. Only go below 1.0 with cause."
                ),
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "watchlist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific things the operator should keep an eye on.",
            },
            "rationale": {"type": "string", "description": "2-4 sentences tying it together."},
        },
        "required": [
            "headline",
            "regime_read",
            "risk_assessment",
            "suggested_risk_posture",
            "confidence",
            "watchlist",
            "rationale",
        ],
    },
}


@dataclass(frozen=True)
class AdvisorBrief:
    headline: str
    regime_read: str
    risk_assessment: str
    suggested_risk_posture: float
    confidence: str
    watchlist: list[str]
    rationale: str

    def render(self) -> str:
        """Slack/markdown-friendly rendering of the brief."""
        posture_note = "full size" if self.suggested_risk_posture >= 0.999 else "suggested DE-RISK"
        lines = [
            f"*{self.headline}*",
            f"- *Regime:* {self.regime_read}",
            f"- *Risk:* {self.risk_assessment}",
            (
                f"- *Suggested posture:* {self.suggested_risk_posture:.2f} ({posture_note}) "
                f"— advisory only, not applied · confidence: {self.confidence}"
            ),
        ]
        if self.watchlist:
            lines.append("- *Watch:* " + "; ".join(self.watchlist))
        lines.append(f"_{self.rationale}_")
        return "\n".join(lines)


def _decision_log_path(data_dir: Path) -> Path:
    return data_dir / "analyst" / "decisions.jsonl"


def _append_decision_log(data_dir: Path, record: dict[str, Any]) -> None:
    """Append one decision to an immutable JSONL audit trail. Best-effort."""
    try:
        path = _decision_log_path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:  # logging must never break the caller
        logger.warning("advisor: failed to write decision log ({!r})", exc)


def _extract_brief(resp: Any) -> AdvisorBrief | None:
    """Pull the submit_brief tool call out of the response. Duck-typed for the
    real SDK and the injected test fake alike."""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == (
            "submit_brief"
        ):
            data = getattr(block, "input", None)
            if not isinstance(data, dict):
                return None
            try:
                posture = float(data["suggested_risk_posture"])
            except (KeyError, TypeError, ValueError):
                posture = 1.0
            posture = max(0.0, min(1.0, posture))  # clamp: advisory, bounded
            watch = data.get("watchlist") or []
            return AdvisorBrief(
                headline=str(data.get("headline", "")).strip(),
                regime_read=str(data.get("regime_read", "")).strip(),
                risk_assessment=str(data.get("risk_assessment", "")).strip(),
                suggested_risk_posture=posture,
                confidence=str(data.get("confidence", "low")).strip(),
                watchlist=[str(w) for w in watch],
                rationale=str(data.get("rationale", "")).strip(),
            )
    return None


def advise(
    facts: str,
    context_text: str,
    *,
    settings: Any,
    asof: date,
    client: Any | None = None,
    data_dir: Path | None = None,
) -> AdvisorBrief | None:
    """Produce a structured advisory brief via Claude. Returns ``None`` when there
    is no API key or on any error (the caller falls back to the digest).

    Read-only: nothing here is ever applied to the live book.
    """
    model = getattr(settings, "anthropic_model", "claude-opus-4-8")
    user_content = (
        "TODAY'S FACTS\n"
        f"{facts}\n\n"
        "RICHER CONTEXT\n"
        f"{context_text}\n\n"
        "Call submit_brief with your structured analysis."
    )
    input_hash = hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16]

    if client is None:
        if not getattr(settings, "anthropic_api_key", None):
            logger.info("advisor: no ANTHROPIC_API_KEY — skipping (digest fallback)")
            return None
        try:
            import anthropic
        except ImportError:
            logger.warning("advisor: anthropic SDK not installed — skipping")
            return None
        # Bounded: short timeout + no retries so a hung call can never hold the
        # shared scheduler 'batch' lock (the dispatcher enforces no timeout).
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=20.0, max_retries=0
        )

    brief: AdvisorBrief | None = None
    error: str | None = None
    # Duck-typed: the same call serves the real SDK and the injected test fake,
    # so type the call target as Any rather than the SDK's strict create() overload.
    api: Any = client
    try:
        # Forced tool_choice guarantees a schema-valid structured output. (We do
        # not enable extended thinking here because a forced single-tool choice is
        # incompatible with it; a future phase can switch to tool_choice=auto +
        # adaptive thinking for deeper regime reasoning.)
        resp = api.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=[
                {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            tools=[_BRIEF_TOOL],
            tool_choice={"type": "tool", "name": "submit_brief"},
            messages=[{"role": "user", "content": user_content}],
        )
        brief = _extract_brief(resp)
        if brief is None:
            error = "no submit_brief tool call in response"
    except Exception as exc:  # never let the daily job die on an API hiccup
        error = repr(exc)
        logger.error("advisor: Claude brief failed ({!r}) — digest fallback", exc)

    if data_dir is not None:
        _append_decision_log(
            data_dir,
            {
                "at": datetime.now(UTC).isoformat(),
                "asof": asof.isoformat(),
                "model": model,
                "input_hash": input_hash,
                "phase": "A-advisory",
                "applied": False,  # Phase A NEVER applies anything
                "brief": asdict(brief) if brief is not None else None,
                "error": error,
            },
        )
    return brief


# --------------------------------------------------------------------------
# Phase B: advise-and-log structured PROPOSALS.
#
# Phase A produces a narrative brief with one advisory posture. Phase B asks for
# concrete, structured DECISIONS — a one-way de-risk throttle, per-strategy
# allocation tilts, and a halt recommendation — then runs them through the
# DETERMINISTIC governance clamp before logging. Crucially it still APPLIES
# NOTHING: the point of Phase B is to prove, over many days of shadow logging,
# that Claude's proposals are sane AND that the clamp reliably neuters anything
# that would violate governance (tilting a non-LIVE strategy, raising risk,
# out-of-range throttle). Promotion to "actually apply the de-risk" is Phase C.
# --------------------------------------------------------------------------

_PROPOSAL_TOOL: dict[str, Any] = {
    "name": "submit_proposals",
    "description": (
        "Submit structured, advisory trading proposals. These are LOGGED and "
        "clamped by governance; none are applied. This is the only allowed output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_throttle": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "One-way DE-RISK multiplier in [0,1]. 1.0 = no change; <1.0 = "
                    "recommend scaling exposure down. You may NEVER recommend >1.0."
                ),
            },
            "allocation_tilt": {
                "type": "array",
                "description": "Per-LIVE-strategy weight deltas (advisory). Only name LIVE strategies.",
                "items": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string"},
                        "delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    },
                    "required": ["slug", "delta"],
                },
            },
            "should_halt": {"type": "boolean", "description": "Recommend a trading halt?"},
            "halt_reason": {"type": "string"},
            "anomaly": {"type": "string", "description": "Any anomaly to triage, else empty."},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string"},
        },
        "required": [
            "risk_throttle",
            "allocation_tilt",
            "should_halt",
            "halt_reason",
            "anomaly",
            "confidence",
            "rationale",
        ],
    },
}

_PROPOSAL_SYSTEM = """\
You are the risk officer for a personal systematic PAPER-trading system. Given the \
day's facts + context, you submit STRUCTURED, ADVISORY proposals via submit_proposals.

Authority is ASYMMETRIC and you must respect it precisely:
- You may only ever recommend REDUCING risk: risk_throttle in [0,1] (never >1.0).
- allocation_tilt may reference ONLY strategies that are currently governance-LIVE \
(shown in context). Tilts naming non-LIVE strategies are discarded by governance.
- You may RECOMMEND a halt; you can never recommend resuming one.
- Nothing you propose is applied automatically — it is clamped by governance and logged.

Be conservative. Default to risk_throttle=1.0 and no halt unless the regime, drawdown, \
guardrails, or evidence give a concrete reason. Ground everything in the input."""


@dataclass(frozen=True)
class Proposals:
    risk_throttle: float  # clamped to [0,1]
    allocation_tilt: dict[str, float]  # LIVE slugs only, after the governance clamp
    dropped_tilts: list[str]  # slugs Claude named that are NOT live (discarded)
    should_halt: bool
    halt_reason: str
    anomaly: str
    confidence: str
    rationale: str

    def render(self) -> str:
        bits = ["*Risk officer (advisory — nothing applied)*"]
        posture = "no de-risk" if self.risk_throttle >= 0.999 else "DE-RISK"
        bits.append(f"- *Throttle:* {self.risk_throttle:.2f} ({posture})")
        if self.allocation_tilt:
            tilt = ", ".join(f"{k} {v:+.0%}" for k, v in sorted(self.allocation_tilt.items()))
            bits.append(f"- *Tilt (live only):* {tilt}")
        if self.dropped_tilts:
            bits.append(f"- *Discarded (not live):* {', '.join(sorted(self.dropped_tilts))}")
        if self.should_halt:
            bits.append(f"- *⚠️ Halt recommended:* {self.halt_reason}")
        if self.anomaly:
            bits.append(f"- *Anomaly:* {self.anomaly}")
        bits.append(f"_{self.rationale}_ (confidence: {self.confidence})")
        return "\n".join(bits)


def _clamp_proposals(data: dict[str, Any], live_slugs: list[str]) -> Proposals:
    """Apply the deterministic governance clamp to Claude's raw proposal.

    This is the safety boundary: out-of-range throttle is bounded one-way, and
    tilts naming non-LIVE strategies are DISCARDED no matter what Claude said.
    """
    live = set(live_slugs)
    try:
        throttle = float(data.get("risk_throttle", 1.0))
    except (TypeError, ValueError):
        throttle = 1.0
    throttle = max(0.0, min(1.0, throttle))  # one-way de-risk only

    tilt: dict[str, float] = {}
    dropped: list[str] = []
    for entry in data.get("allocation_tilt", []) or []:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug", "")).strip()
        if not slug:
            continue
        try:
            delta = float(entry.get("delta", 0.0))
        except (TypeError, ValueError):
            continue
        if slug in live:
            tilt[slug] = max(-1.0, min(1.0, delta))
        else:
            dropped.append(slug)  # governance discards non-live tilts

    return Proposals(
        risk_throttle=throttle,
        allocation_tilt=tilt,
        dropped_tilts=dropped,
        should_halt=bool(data.get("should_halt", False)),
        halt_reason=str(data.get("halt_reason", "")).strip(),
        anomaly=str(data.get("anomaly", "")).strip(),
        confidence=str(data.get("confidence", "low")).strip(),
        rationale=str(data.get("rationale", "")).strip(),
    )


def _extract_proposals(resp: Any, live_slugs: list[str]) -> Proposals | None:
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == (
            "submit_proposals"
        ):
            data = getattr(block, "input", None)
            if not isinstance(data, dict):
                return None
            return _clamp_proposals(data, live_slugs)
    return None


def propose(
    facts: str,
    context_text: str,
    *,
    settings: Any,
    asof: date,
    live_slugs: list[str],
    client: Any | None = None,
    data_dir: Path | None = None,
    model: str | None = None,
) -> Proposals | None:
    """Phase B: structured advisory proposals, governance-clamped and logged.

    Applies NOTHING. Returns ``None`` with no key / on any error. ``model`` lets a
    caller (e.g. the intraday shadow log) override the default with a cheaper one.
    """
    model = model or getattr(settings, "anthropic_model", "claude-opus-4-8")
    user_content = (
        "TODAY'S FACTS\n"
        f"{facts}\n\n"
        "RICHER CONTEXT\n"
        f"{context_text}\n\n"
        f"GOVERNANCE-LIVE STRATEGIES (only these may be tilted): {live_slugs or '(none)'}\n\n"
        "Call submit_proposals with your structured, conservative proposals."
    )
    input_hash = hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16]

    if client is None:
        if not getattr(settings, "anthropic_api_key", None):
            logger.info("advisor.propose: no ANTHROPIC_API_KEY — skipping")
            return None
        try:
            import anthropic
        except ImportError:
            logger.warning("advisor.propose: anthropic SDK not installed — skipping")
            return None
        # Bounded: short timeout + no retries so a hung call can never hold the
        # shared scheduler 'batch' lock (the dispatcher enforces no timeout).
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=20.0, max_retries=0
        )

    proposals: Proposals | None = None
    error: str | None = None
    api: Any = client
    try:
        resp = api.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=[
                {"type": "text", "text": _PROPOSAL_SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            tools=[_PROPOSAL_TOOL],
            tool_choice={"type": "tool", "name": "submit_proposals"},
            messages=[{"role": "user", "content": user_content}],
        )
        proposals = _extract_proposals(resp, live_slugs)
        if proposals is None:
            error = "no submit_proposals tool call in response"
    except Exception as exc:
        error = repr(exc)
        logger.error("advisor.propose: Claude proposal failed ({!r})", exc)

    if data_dir is not None:
        _append_decision_log(
            data_dir,
            {
                "at": datetime.now(UTC).isoformat(),
                "asof": asof.isoformat(),
                "model": model,
                "input_hash": input_hash,
                "phase": "B-advise-and-log",
                "applied": False,  # Phase B NEVER applies anything
                "live_slugs": list(live_slugs),
                "proposals": asdict(proposals) if proposals is not None else None,
                "error": error,
            },
        )
    return proposals
