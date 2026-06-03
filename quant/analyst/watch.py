"""Intraday, READ-ONLY Claude analyst watch (Sprint 1 of the continuous analyst).

Where the daily ``digest``/``brief`` run once after the close, the *watch* lets
Claude comment on the live book a few times DURING the trading day (scheduled
slots: open / midday / power-hour). Each run gathers the same read-only facts the
digest uses, asks Claude for a short structured intraday note, posts it to Slack,
and records the call on the immutable ``decisions.jsonl`` audit trail.

Hard safety boundaries (identical envelope to the Phase-A brief — verified by
``tests/analyst/test_watch.py``):

* APPLIES NOTHING. No order path, no halt, no allocation change. The Claude tool
  schema carries NO actionable fields (no throttle/tilt/halt) — only narrative.
* FAIL-OPEN. Any error / missing key degrades to a deterministic template (or no
  post) and never raises, so a watch run can never crash the launchd tick.
* BOUNDED. The Anthropic client uses a short timeout + zero retries (a hung call
  must never hold the shared scheduler ``batch`` lock — ``max_runtime_s`` is not
  enforced by the dispatcher), and a per-session post cap + min-gap are enforced
  BEFORE the Claude call so cost/Slack volume stay bounded even under a thrash.
* NON-SPAMMY. Identical commentary is not re-posted; suppression is appended to
  the audit log, never silent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant.analyst.advisor import propose
from quant.analyst.digest import DigestData, gather_digest_data, render_facts
from quant.util.atomic import write_json_atomic
from quant.util.logging import logger

_MAX_TOKENS = 500
_DEFAULT_MAX_POSTS = 4
_DEFAULT_MIN_GAP_MIN = 30
# Hard client-side ceiling: the dispatcher runs jobs via subprocess.run with NO
# timeout and ``max_runtime_s`` is parsed but never enforced, so this is the only
# thing bounding how long a hung Claude call holds the shared 'batch' lock.
_CLIENT_TIMEOUT_S = 20.0
_PHASE = "watch-intraday"

_SYSTEM_PROMPT = """\
You are the in-house analyst for a personal systematic PAPER-trading system, \
giving a SHORT intraday read during market hours (the slot — open/midday/power-hour \
— is provided). You receive the day's read-only facts plus a richer context pack \
(regime + posteriors, macro, per-strategy validation/governance, portfolio risk). \
Write a brief, decision-useful intraday note for the operator's phone.

You are ADVISORY ONLY and intraday: you never place orders, never halt, never \
change allocation. ``posture_note`` is a single word (steady/watch/caution), not a \
number and not an action — nothing you write is applied.

Rules:
- Ground every claim in the provided facts/context. Never invent numbers or strategies.
- Respect governance: only the strategy shown LIVE is authorized; quarantined \
strategies are not eligible no matter how you read them.
- This is PAPER trading and may be in DRY-RUN; never imply real money is at stake.
- Be terse: a headline plus one or two lines. Submit your answer ONLY by calling \
the ``submit_commentary`` tool."""

# The schema deliberately carries NO actionable fields (no risk_throttle, no
# allocation_tilt, no should_halt) — so even a misbehaving model cannot encode a
# trade or a halt. posture_note is a narrative enum, not a number.
_WATCH_TOOL: dict[str, Any] = {
    "name": "submit_commentary",
    "description": (
        "Submit a short intraday analyst note. This is the only allowed output and "
        "is ADVISORY — nothing here is ever applied to the live book."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": "One short line — the intraday bottom line for the operator's phone.",
            },
            "whats_moving": {
                "type": "string",
                "description": "What is notable / has moved this session, grounded in the facts.",
            },
            "posture_note": {
                "type": "string",
                "enum": ["steady", "watch", "caution"],
                "description": "Narrative posture only — a word, NOT a number, NOT applied.",
            },
            "watchlist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific things to keep an eye on into the close.",
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["headline", "whats_moving", "posture_note", "watchlist", "confidence"],
    },
}


@dataclass(frozen=True)
class WatchComment:
    headline: str
    whats_moving: str
    posture_note: str  # one of: steady | watch | caution
    watchlist: list[str]
    confidence: str

    def render(self, slot: str) -> str:
        """Slack/markdown-friendly rendering of the intraday note."""
        note = {"steady": "steady", "watch": "watching", "caution": "⚠️ caution"}.get(
            self.posture_note, self.posture_note
        )
        lines = [
            f"*[{slot}] {self.headline}*",
            f"- {self.whats_moving}",
            f"- *Posture:* {note} (advisory only, not applied) · confidence: {self.confidence}",
        ]
        if self.watchlist:
            lines.append("- *Watch:* " + "; ".join(self.watchlist))
        return "\n".join(lines)


@dataclass(frozen=True)
class WatchResult:
    body: str | None
    used_llm: bool
    posted: bool
    suppressed_reason: str | None


@dataclass(frozen=True)
class _WatchState:
    session: str  # asof.isoformat()
    posts_today: int
    last_post_at: datetime | None
    last_hash: str | None


# --------------------------------------------------------------------------
# Audit log + per-session state (both best-effort / fail-open).
# --------------------------------------------------------------------------


def _append_decision(data_dir: Path, record: dict[str, Any]) -> None:
    """Append one watch event to the immutable JSONL audit trail. Best-effort."""
    try:
        path = data_dir / "analyst" / "decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:  # logging must never break the caller
        logger.warning("watch: failed to write decision log ({!r})", exc)


def _watch_state_path(data_dir: Path) -> Path:
    return data_dir / "analyst" / "watch_state.json"


def load_watch_state(data_dir: Path, asof: date) -> _WatchState:
    """Read the per-session post counter/fingerprint. Fail-open to a fresh state
    (a corrupt/missing/older-session file resets to 0 posts — at worst one extra
    benign post, never a crash)."""
    empty = _WatchState(asof.isoformat(), 0, None, None)
    try:
        path = _watch_state_path(data_dir)
        if not path.exists():
            return empty
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("session") != asof.isoformat():
            return empty  # a new session resets the counters
        lpa = raw.get("last_post_at")
        return _WatchState(
            session=asof.isoformat(),
            posts_today=int(raw.get("posts_today", 0)),
            last_post_at=datetime.fromisoformat(lpa) if lpa else None,
            last_hash=raw.get("last_hash"),
        )
    except Exception as exc:
        logger.warning("watch: failed to read state ({!r}) — treating as fresh", exc)
        return empty


def save_watch_state(data_dir: Path, state: _WatchState) -> None:
    """Atomically persist the per-session counter/fingerprint. Best-effort."""
    try:
        write_json_atomic(
            _watch_state_path(data_dir),
            {
                "session": state.session,
                "posts_today": state.posts_today,
                "last_post_at": state.last_post_at.isoformat() if state.last_post_at else None,
                "last_hash": state.last_hash,
            },
        )
    except Exception as exc:
        logger.warning("watch: failed to save state ({!r})", exc)


# --------------------------------------------------------------------------
# Claude call (bounded, forced-tool, fail-open) + deterministic fallback.
# --------------------------------------------------------------------------


def _extract_comment(resp: Any) -> WatchComment | None:
    """Pull the submit_commentary tool call out of the response. Duck-typed for
    the real SDK and the injected test fake alike."""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == (
            "submit_commentary"
        ):
            data = getattr(block, "input", None)
            if not isinstance(data, dict):
                return None
            posture = str(data.get("posture_note", "steady")).strip()
            if posture not in {"steady", "watch", "caution"}:
                posture = "steady"
            watch = data.get("watchlist") or []
            return WatchComment(
                headline=str(data.get("headline", "")).strip(),
                whats_moving=str(data.get("whats_moving", "")).strip(),
                posture_note=posture,
                watchlist=[str(w) for w in watch],
                confidence=str(data.get("confidence", "low")).strip(),
            )
    return None


def comment(
    facts: str,
    context_text: str,
    *,
    settings: Any,
    asof: date,
    slot: str,
    client: Any | None = None,
    data_dir: Path | None = None,
) -> WatchComment | None:
    """Produce a short structured intraday note via Claude. Returns ``None`` when
    there is no API key or on any error (the caller falls back to the template).

    Read-only: nothing here is ever applied to the live book.
    """
    # Routine intraday summarization runs on the cheaper "fast" model when set
    # (Opus is reserved for the daily brief / weekly synthesis) to keep cost low.
    model = getattr(settings, "anthropic_model_fast", None) or getattr(
        settings, "anthropic_model", "claude-opus-4-8"
    )
    user_content = (
        f"SLOT: {slot}\n\n"
        "TODAY'S FACTS\n"
        f"{facts}\n\n"
        "RICHER CONTEXT\n"
        f"{context_text}\n\n"
        "Call submit_commentary with a short intraday note."
    )
    input_hash = hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16]

    if client is None:
        if not getattr(settings, "anthropic_api_key", None):
            logger.info("watch: no ANTHROPIC_API_KEY — template fallback")
            return None
        try:
            import anthropic
        except ImportError:
            logger.warning("watch: anthropic SDK not installed — template fallback")
            return None
        # Bounded: a short timeout + NO retries so a hung call can never hold the
        # shared scheduler 'batch' lock (the dispatcher enforces no timeout).
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=_CLIENT_TIMEOUT_S,
            max_retries=0,
        )

    cmt: WatchComment | None = None
    error: str | None = None
    api: Any = client
    try:
        resp = api.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=[
                {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            tools=[_WATCH_TOOL],
            tool_choice={"type": "tool", "name": "submit_commentary"},
            messages=[{"role": "user", "content": user_content}],
        )
        cmt = _extract_comment(resp)
        if cmt is None:
            error = "no submit_commentary tool call in response"
    except Exception as exc:  # never let an intraday job die on an API hiccup
        error = repr(exc)
        logger.error("watch: Claude commentary failed ({!r}) — template fallback", exc)

    if data_dir is not None:
        _append_decision(
            data_dir,
            {
                "at": datetime.now(UTC).isoformat(),
                "asof": asof.isoformat(),
                "model": model,
                "input_hash": input_hash,
                "phase": _PHASE,
                "applied": False,  # the watch NEVER applies anything
                "slot": slot,
                "comment": asdict(cmt) if cmt is not None else None,
                "error": error,
            },
        )
    return cmt


def render_watch(d: DigestData, slot: str) -> str:
    """Deterministic plain-text intraday note — the no-LLM fallback body."""
    lines = [f"*[{slot}] intraday read* — {d.asof.isoformat()}"]
    if d.equity is not None:
        pct = d.day_pl_pct
        pct_str = f" ({pct:+.2%})" if pct is not None else ""
        lines.append(f"- Equity ${d.equity:,.0f}{pct_str}")
    else:
        lines.append("- Equity: no snapshot available")
    sev = d.guard_worst_severity or "unknown"
    halt = " · HALT ACTIVE" if d.halt_active else ""
    lines.append(f"- Guardrails: {sev}{halt}")
    lines.append("- Live: " + (", ".join(d.governance_live) if d.governance_live else "(none)"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration: gather → (hard gate) → Claude → de-dup → Slack → state.
# --------------------------------------------------------------------------


def _slack_text(asof: date, slot: str, body: str) -> str:
    return f"🟢 quant intraday [{slot}] — {asof.isoformat()}\n\n{body}"


def _slack_blocks(asof: date, slot: str, body: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🟢 quant intraday [{slot}] — {asof.isoformat()}"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": body[:2900]}},
    ]


def run_watch(
    *,
    data_dir: Path,
    asof: date,
    settings: Any,
    alerts: Any,
    slot: str = "midday",
    now: datetime | None = None,
    client: Any | None = None,
    dry_run: bool = False,
    account: dict[str, float] | None = None,
    live_positions: list[tuple[str, int]] | None = None,
    governance_live: list[str] | None = None,
    halt_active: bool = False,
    context_text: str | None = None,
    shadow_proposals: bool = True,
    max_posts: int = _DEFAULT_MAX_POSTS,
    min_gap_min: int = _DEFAULT_MIN_GAP_MIN,
) -> WatchResult:
    """Gather facts, ask Claude for a short intraday note, and post to Slack.

    Cost/spam are bounded BEFORE any Claude call: a per-session post cap and a
    min-gap short-circuit early; identical commentary is de-duplicated. Every
    path — post or suppression — is recorded on the audit log.
    """
    now = now or datetime.now(UTC)
    state = load_watch_state(data_dir, asof)

    # HARD pre-Claude gates: bound cost + Slack volume regardless of the model.
    if state.posts_today >= max_posts:
        reason = f"daily cap reached ({state.posts_today}/{max_posts})"
        _append_decision(
            data_dir,
            {"at": now.isoformat(), "asof": asof.isoformat(), "phase": _PHASE,
             "applied": False, "slot": slot, "suppressed": reason},
        )
        return WatchResult(body=None, used_llm=False, posted=False, suppressed_reason=reason)
    if state.last_post_at is not None and (now - state.last_post_at).total_seconds() < min_gap_min * 60:
        reason = f"within min-gap ({min_gap_min}m)"
        _append_decision(
            data_dir,
            {"at": now.isoformat(), "asof": asof.isoformat(), "phase": _PHASE,
             "applied": False, "slot": slot, "suppressed": reason},
        )
        return WatchResult(body=None, used_llm=False, posted=False, suppressed_reason=reason)

    facts_data = gather_digest_data(
        data_dir,
        asof,
        dry_run=dry_run,
        account=account,
        live_positions=live_positions,
        governance_live=governance_live,
        halt_active=halt_active,
    )
    facts = render_facts(facts_data)
    cmt = comment(
        facts,
        context_text or "(no additional context)",
        settings=settings,
        asof=asof,
        slot=slot,
        client=client,
        data_dir=data_dir,
    )
    used_llm = cmt is not None

    # Shadow-log a governance-clamped one-way de-risk PROPOSAL (Phase B) on the
    # cheap model — it APPLIES NOTHING and posts nothing, it only appends to the
    # decisions.jsonl audit trail. This builds the multi-week bake-in evidence the
    # roadmap requires before any (human-gated) one-way de-risk actuator (Phase C).
    if shadow_proposals and governance_live:
        try:
            fast_model = getattr(settings, "anthropic_model_fast", None) or getattr(
                settings, "anthropic_model", "claude-opus-4-8"
            )
            propose(
                facts,
                context_text or "(no additional context)",
                settings=settings,
                asof=asof,
                live_slugs=list(governance_live),
                client=client,
                data_dir=data_dir,
                model=fast_model,
            )
        except Exception as exc:  # fail-open: shadow logging must never break the watch
            logger.warning("watch: shadow proposal failed ({!r})", exc)

    body = cmt.render(slot) if cmt is not None else render_watch(facts_data, slot)
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    if state.last_hash is not None and body_hash == state.last_hash:
        reason = "duplicate content"
        _append_decision(
            data_dir,
            {"at": now.isoformat(), "asof": asof.isoformat(), "phase": _PHASE,
             "applied": False, "slot": slot, "suppressed": reason},
        )
        return WatchResult(body=body, used_llm=used_llm, posted=False, suppressed_reason=reason)

    posted = False
    if not dry_run:
        try:
            posted = bool(
                alerts.send_slack(_slack_text(asof, slot, body), blocks=_slack_blocks(asof, slot, body))
            )
        except Exception as exc:  # fail-open: a Slack hiccup must never crash a tick
            logger.error("watch: Slack post failed ({!r})", exc)
            posted = False

    if posted:
        save_watch_state(
            data_dir,
            _WatchState(
                session=asof.isoformat(),
                posts_today=state.posts_today + 1,
                last_post_at=now,
                last_hash=body_hash,
            ),
        )

    return WatchResult(body=body, used_llm=used_llm, posted=posted, suppressed_reason=None)
