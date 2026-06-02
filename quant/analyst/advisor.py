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
        posture_note = (
            "full size" if self.suggested_risk_posture >= 0.999 else "suggested DE-RISK"
        )
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
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

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
