"""The continuous, always-on engine loop (Phase 6).

Every cycle: read the live book (best-effort), build a ``MarketState``, persist
it (hot snapshot + append-only audit + heartbeat), detect material events, post
them to Slack, and — only for the highest-severity events, rate-limited and
cost-capped — ask Claude for a one-line interpretation. It NEVER trades, halts,
or changes governance/allocation. Fail-safe: any per-cycle error is logged and
the loop continues; the cadence is market-hours-aware (tight in RTH, slow when
closed). All seams (clock, sleep, broker, Slack, Claude) are injectable for tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant.deploy.calendar_clock import to_et
from quant.engine.events import EngineEvent, EventConfig, detect_events, severity_at_least
from quant.engine.intraday import live_intraday_signals
from quant.engine.state import (
    MarketState,
    build_market_state,
    render_state,
    session_phase,
    to_json_dict,
)
from quant.fundamentals.factors import live_fundamentals
from quant.macro.events import live_event_risk
from quant.macro.nowcast import live_macro_nowcast
from quant.nlp.sentiment import live_news_sentiment
from quant.util.atomic import write_json_atomic
from quant.util.logging import logger


@dataclass(frozen=True)
class EngineConfig:
    """Loop cadence + Claude cost controls. Defaults err toward cheap & quiet."""

    cadence_rth_s: float = 45.0
    cadence_premarket_s: float = 120.0
    cadence_offhours_s: float = 600.0
    # Claude escalation is for IMPACTFUL events only, and is hard cost-capped.
    claude_severity: str = "critical"  # minimum severity that may reach Claude
    claude_min_gap_s: float = 1800.0  # >= 30 min between Claude calls (survives restarts)
    claude_max_per_session: int = 6
    # An event whose code already fired within this window is suppressed (no Slack
    # spam, no re-escalation) — a persistent condition reports once, not every cycle.
    event_dedup_window_s: float = 3600.0
    # News sentiment refreshes on a slower cadence than the price snapshot
    # (headlines arrive sporadically) and is reused between refreshes.
    news_refresh_s: float = 180.0
    news_lookback_minutes: int = 240
    # Event-risk (FRED uncertainty + calendar) moves daily at most — refresh slowly.
    eventrisk_refresh_s: float = 1800.0
    # Fundamentals (PIT EDGAR facts) only change on filings; the read moves with
    # price (earnings yield), so refresh a few times a session, not every cycle.
    fundamentals_refresh_s: float = 21600.0  # 6h
    # Macro nowcast (FRED curve/credit/claims/Sahm) updates daily at most.
    nowcast_refresh_s: float = 21600.0  # 6h
    events: EventConfig = field(default_factory=EventConfig)


def engine_dir(data_dir: Path) -> Path:
    return data_dir / "engine"


def _default_positions(settings: Any) -> dict[str, int] | None:
    try:
        from quant.execution.alpaca import AlpacaClient

        rows = AlpacaClient(settings=settings).positions()
        return {str(r.symbol): int(r.qty) for r in rows}
    except Exception as exc:  # broker read is best-effort
        logger.info("engine: positions read skipped ({!r})", exc)
        return None


def _default_equity(settings: Any) -> float | None:
    try:
        from quant.execution.alpaca import AlpacaClient

        return float(AlpacaClient(settings=settings).account().equity)
    except Exception as exc:  # broker read is best-effort
        logger.info("engine: equity read skipped ({!r})", exc)
        return None


def _cadence_s(phase: str, cfg: EngineConfig) -> float:
    if phase == "rth":
        return cfg.cadence_rth_s
    if phase in ("premarket", "afterhours"):
        return cfg.cadence_premarket_s
    return cfg.cadence_offhours_s


def summarize_events(state: MarketState, events: list[EngineEvent], settings: Any) -> str:
    """One-line Claude interpretation of impactful events. Bounded + FAIL-OPEN to
    a deterministic template (so a hung/absent API never blocks or costs)."""
    template = "; ".join(f"{e.code}: {e.detail}" for e in events)
    api_key = getattr(settings, "anthropic_api_key", None)
    if not api_key:
        return template
    try:
        import anthropic

        model = str(
            getattr(settings, "anthropic_model_fast", None)
            or getattr(settings, "anthropic_model", "claude-haiku-4-5")
        )
        client = anthropic.Anthropic(api_key=api_key, timeout=20.0, max_retries=0)
        ev_lines = "\n".join(f"- [{e.severity}] {e.code}: {e.detail}" for e in events)
        prompt = (
            "You are a risk-desk analyst. In ONE sentence (<=40 words), say what these "
            "live-market events mean for a defensive ETF book and whether they warrant "
            "human attention. Be specific and sober; do not invent numbers.\n\n"
            f"Market state: {render_state(state)}\n\nEvents:\n{ev_lines}"
        )
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [getattr(b, "text", "") for b in resp.content]
        text = " ".join(p for p in parts if p).strip()
        return text or template
    except Exception as exc:  # fail-open: the deterministic template still informs
        logger.info("engine: Claude summary skipped ({!r})", exc)
        return template


def _claude_state_path(data_dir: Path) -> Path:
    return engine_dir(data_dir) / "claude_state.json"


def _persisted_last_claude(data_dir: Path) -> datetime | None:
    """Last Claude-escalation time persisted across restarts (crash-loop guard)."""
    try:
        p = _claude_state_path(data_dir)
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return datetime.fromisoformat(str(raw["last_at"])) if raw.get("last_at") else None
    except Exception:
        return None


def _record_claude(data_dir: Path, now: datetime) -> None:
    try:
        write_json_atomic(_claude_state_path(data_dir), {"last_at": now.isoformat()})
    except Exception as exc:  # best-effort
        logger.info("engine: claude_state write skipped ({!r})", exc)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    except Exception as exc:  # audit log is best-effort
        logger.warning("engine: append {} skipped ({!r})", path.name, exc)


def run_engine(
    settings: Any,
    *,
    config: EngineConfig | None = None,
    once: bool = False,
    max_cycles: int | None = None,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
    positions_fn: Callable[[], dict[str, int] | None] | None = None,
    equity_fn: Callable[[], float | None] | None = None,
    intraday_fn: Callable[[], Any] | None = None,
    news_fn: Callable[[], Any] | None = None,
    eventrisk_fn: Callable[[date], Any] | None = None,
    fundamentals_fn: Callable[[date], Any] | None = None,
    macro_nowcast_fn: Callable[[date], Any] | None = None,
    slack: Any | None = None,
    claude_fn: Callable[[MarketState, list[EngineEvent], Any], str] | None = None,
    console_print: Callable[[str], None] | None = None,
) -> list[MarketState]:
    """Run the continuous engine. ``once``/``max_cycles`` bound it for tests/one-shot.

    ``dry_run`` computes + persists state and logs events but posts NOTHING to
    Slack and never calls Claude. The loop actuates no trades under any setting.
    """
    cfg = config or EngineConfig()
    data_dir = Path(settings.data_dir)
    now_fn = now_fn or (lambda: datetime.now(UTC))
    pos_fn = positions_fn or (lambda: _default_positions(settings))
    eq_fn = equity_fn or (lambda: _default_equity(settings))
    intra_fn = intraday_fn or (lambda: live_intraday_signals(settings))
    news_fn_ = news_fn or (
        lambda: live_news_sentiment(settings, lookback_minutes=cfg.news_lookback_minutes)
    )
    er_fn = eventrisk_fn or (lambda d: live_event_risk(settings, d))
    fund_fn = fundamentals_fn or (lambda d: live_fundamentals(settings, d))
    nowcast_fn = macro_nowcast_fn or (lambda d: live_macro_nowcast(settings, d))
    claude = claude_fn or summarize_events
    if slack is None and not dry_run:
        from quant.deploy.alerts import AlertClient, AlertConfig

        slack = AlertClient(
            AlertConfig(
                healthcheck_tick_url=getattr(settings, "healthcheck_tick_url", None),
                healthcheck_guard_url=getattr(settings, "healthcheck_guard_url", None),
                pushover_app_token=getattr(settings, "pushover_app_token", None),
                pushover_user_key=getattr(settings, "pushover_user_key", None),
                slack_webhook_url=getattr(settings, "slack_webhook_url", None),
            )
        )

    states: list[MarketState] = []
    prev: MarketState | None = None
    session_date: str | None = None
    session_high_equity: float | None = None
    last_event_at: dict[str, datetime] = {}
    last_claude_at: datetime | None = _persisted_last_claude(data_dir)
    claude_session_count = 0
    last_news_at: datetime | None = None
    cached_news: Any = None
    last_er_at: datetime | None = None
    cached_er: Any = None
    last_fund_at: datetime | None = None
    cached_fund: Any = None
    last_nowcast_at: datetime | None = None
    cached_nowcast: Any = None
    cycle = 0

    while True:
        phase = "closed"
        try:
            now = now_fn()
            asof = to_et(now).date()
            phase = session_phase(now, asof)
            positions = pos_fn()
            equity = eq_fn()
            # Intraday snapshot only while the tape is active (skip overnight calls).
            intraday = intra_fn() if phase != "closed" else None
            # News sentiment on its own slower cadence; reused between refreshes.
            if (
                cached_news is None
                or last_news_at is None
                or (now - last_news_at).total_seconds() >= cfg.news_refresh_s
            ):
                cached_news = news_fn_()
                last_news_at = now
            # Event risk (FRED uncertainty + scheduled-event calendar): slow cadence.
            if (
                cached_er is None
                or last_er_at is None
                or (now - last_er_at).total_seconds() >= cfg.eventrisk_refresh_s
            ):
                cached_er = er_fn(asof)
                last_er_at = now
            # Fundamentals (PIT EDGAR + price): slow cadence, reused between refreshes.
            if (
                cached_fund is None
                or last_fund_at is None
                or (now - last_fund_at).total_seconds() >= cfg.fundamentals_refresh_s
            ):
                cached_fund = fund_fn(asof)
                last_fund_at = now
            # Macro nowcast (FRED curve/credit/claims/Sahm): slow cadence.
            if (
                cached_nowcast is None
                or last_nowcast_at is None
                or (now - last_nowcast_at).total_seconds() >= cfg.nowcast_refresh_s
            ):
                cached_nowcast = nowcast_fn(asof)
                last_nowcast_at = now
            state = build_market_state(
                data_dir,
                asof=asof,
                now_utc=now,
                positions=positions,
                equity=equity,
                intraday=intraday,
                news=cached_news,
                event_risk=cached_er,
                fundamentals=cached_fund,
                macro_nowcast=cached_nowcast,
            )

            # Session anchor (resets each ET trading date) for intraday drawdown.
            if state.asof != session_date:
                session_date = state.asof
                session_high_equity = state.equity
                claude_session_count = 0
            if state.equity is not None:
                session_high_equity = (
                    state.equity
                    if session_high_equity is None
                    else max(session_high_equity, state.equity)
                )

            # Persist: hot snapshot + append-only audit + heartbeat.
            payload = to_json_dict(state)
            edir = engine_dir(data_dir)
            write_json_atomic(edir / "state.json", payload)
            _append_jsonl(edir / "state.jsonl", payload)
            write_json_atomic(
                edir / "heartbeat.json",
                {"at": state.at, "cycle": cycle, "phase": phase, "degraded": list(state.degraded)},
            )

            # Detect, dedup, notify.
            events = detect_events(prev, state, cfg.events, session_high_equity=session_high_equity)
            fresh = [
                e
                for e in events
                if e.code not in last_event_at
                or (now - last_event_at[e.code]).total_seconds() >= cfg.event_dedup_window_s
            ]
            for e in fresh:
                last_event_at[e.code] = now
                _append_jsonl(
                    edir / "events.jsonl",
                    {"at": e.at, "code": e.code, "severity": e.severity, "detail": e.detail},
                )
                if not dry_run and slack is not None:
                    slack.send_slack(f":satellite: *engine* [{e.severity}] {e.code}: {e.detail}")

            # Escalate ONLY impactful, fresh events to Claude — rate-limited + capped.
            impactful = [e for e in fresh if severity_at_least(e.severity, cfg.claude_severity)]
            gap_ok = last_claude_at is None or (
                (now - last_claude_at).total_seconds() >= cfg.claude_min_gap_s
            )
            if (
                impactful
                and not dry_run
                and gap_ok
                and claude_session_count < cfg.claude_max_per_session
            ):
                summary = claude(state, impactful, settings)
                last_claude_at = now
                claude_session_count += 1
                _record_claude(data_dir, now)
                if slack is not None:
                    slack.send_slack(f":brain: *engine analysis*: {summary}")

            if console_print is not None:
                console_print(render_state(state) + (f"  ({len(fresh)} events)" if fresh else ""))
            states.append(state)
            prev = state
        except Exception as exc:  # fail-safe: never crash the daemon
            logger.warning("engine cycle error (continuing): {!r}", exc)
            if console_print is not None:
                console_print(f"engine cycle error (continuing): {exc!r}")

        cycle += 1
        if once or (max_cycles is not None and cycle >= max_cycles):
            break
        sleep(_cadence_s(phase, cfg))

    return states
