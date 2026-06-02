"""Daily analyst digest: gather → narrate (Claude) → deliver (Slack).

Design:
* ``gather_digest_data`` is pure-ish — it reads files under ``data_dir`` and the
  optional live Alpaca snapshot the caller injects, and returns a plain dataclass.
  No network, no Claude, no Slack — so it is trivially unit-testable.
* ``render_facts`` turns that dataclass into a deterministic plain-text block.
  This doubles as the LLM input AND the no-LLM fallback body, so the digest is
  always useful even with no API key.
* ``narrate`` calls the Claude API (injectable client for tests) and returns the
  narrative, or ``None`` on any failure / missing key — the caller falls back to
  the deterministic facts. A daily job must never crash because an LLM call did.
* ``run_digest`` orchestrates, writes a committed markdown artifact under
  ``docs/analyst/``, and delivers to Slack (unless ``dry_run``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant.live.bookkeeping import read_equity, read_trades
from quant.util.logging import logger

# Narration is a short, templated summarization of structured facts — not an
# open-ended reasoning task — so we run the model WITHOUT thinking (which on
# Opus 4.8 means a plain, fast response) and lean on the system prompt for
# concision. The model ID comes from Settings (defaults to claude-opus-4-8).
_SYSTEM_PROMPT = """\
You are the in-house analyst for a personal systematic PAPER-trading system. \
Each evening you receive a structured summary of one trading day and write a \
short, plain-English digest for the operator's phone (Slack).

Style:
- Lead with the single most important thing: P&L, a halt, a missed trade, or "quiet day".
- 4-8 short sentences or compact bullets. No preamble, no sign-off, no meta-commentary.
- Be precise with the numbers given; never invent data that is not in the input.
- This is PAPER trading and may be in DRY-RUN mode — say so plainly when the input \
flags it, and never imply real money is at stake.
- Plain text with light Slack markdown only (*bold*, "- " bullets). No headings, no code fences.

Respond with ONLY the digest text."""

_MAX_TOKENS = 1024


@dataclass(frozen=True)
class DigestData:
    """Structured facts for one trading day. All primitives — safe to render/serialize."""

    asof: date
    dry_run: bool
    equity: float | None
    prev_equity: float | None
    cash: float | None
    governance_live: list[str]
    positions: list[tuple[str, int]]  # (symbol, qty)
    orders: list[dict[str, Any]]  # {strategy, symbol, side, qty, dry_run}
    guard_worst_severity: str | None
    guard_heartbeat: str | None
    guard_outcomes: list[tuple[str, str]]  # (name, severity)
    halt_active: bool
    jobs: list[tuple[str, str, int]] = field(default_factory=list)  # (job, kind, exit_code)

    @property
    def day_pl(self) -> float | None:
        if self.equity is None or self.prev_equity is None:
            return None
        return self.equity - self.prev_equity

    @property
    def day_pl_pct(self) -> float | None:
        pl = self.day_pl
        if pl is None or not self.prev_equity:
            return None
        return pl / self.prev_equity


@dataclass(frozen=True)
class DigestResult:
    body: str
    facts: str
    used_llm: bool
    delivered: bool
    artifact_path: Path | None


def _latest_positions_snapshot(data_dir: Path) -> list[tuple[str, int]]:
    """Aggregate the most-recent ``strategy_positions`` snapshot by symbol (fallback)."""
    path = data_dir / "live" / "strategy_positions.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if df.empty or "date" not in df.columns:
        return []
    latest = df[df["date"] == df["date"].max()]
    agg = latest.groupby("symbol")["qty"].sum()
    return [(str(sym), int(qty)) for sym, qty in agg.items() if int(qty) != 0]


def _read_guard_status(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / "ops" / "monitor_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, ValueError):
        return None


def _read_jobs(data_dir: Path, asof: date) -> list[tuple[str, str, int]]:
    """Read today's scheduler run-ledger markers (job, kind, exit_code)."""
    sched = data_dir / "ops" / "scheduler"
    if not sched.exists():
        return []
    out: list[tuple[str, str, int]] = []
    suffix = f".{asof.isoformat()}.json"
    for marker in sorted(sched.glob(f"*{suffix}")):
        try:
            m = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append((str(m.get("job", marker.name)), str(m.get("kind", "?")), int(m.get("exit_code", -1))))
    return out


def gather_digest_data(
    data_dir: Path,
    asof: date,
    *,
    dry_run: bool = False,
    account: dict[str, float] | None = None,
    live_positions: list[tuple[str, int]] | None = None,
    governance_live: list[str] | None = None,
    halt_active: bool = False,
) -> DigestData:
    """Collect one day's facts. ``account``/``live_positions`` are the live Alpaca
    snapshot the caller injects (preferred); otherwise we fall back to the cached
    parquet books so the digest still works offline / in tests."""
    # Equity: prefer the live Alpaca account; fall back to the latest equity row.
    equity = prev_equity = cash = None
    if account is not None:
        equity = account.get("equity")
        prev_equity = account.get("last_equity")
        cash = account.get("cash")
    else:
        eq = read_equity(data_dir)
        if not eq.empty:
            last = eq.iloc[-1]
            equity = float(last["equity"])
            prev_equity = float(last["last_equity"]) if "last_equity" in eq.columns else None
            cash = float(last["cash"]) if "cash" in eq.columns else None

    positions = live_positions if live_positions is not None else _latest_positions_snapshot(data_dir)

    # Orders submitted (or would-be, in dry-run) for asof.
    orders: list[dict[str, Any]] = []
    trades = read_trades(data_dir)
    if not trades.empty:
        today = trades[trades["date"] == pd.Timestamp(asof)]
        for _, row in today.iterrows():
            orders.append(
                {
                    "strategy": str(row["strategy"]),
                    "symbol": str(row["symbol"]),
                    "side": str(row["side"]),
                    "qty": int(row["qty"]),
                    "dry_run": bool(row["dry_run"]),
                }
            )

    guard = _read_guard_status(data_dir)
    guard_outcomes: list[tuple[str, str]] = []
    guard_worst = guard_heartbeat = None
    if guard is not None:
        guard_worst = guard.get("worst_severity")
        guard_heartbeat = guard.get("heartbeat")
        for o in guard.get("outcomes", []):
            guard_outcomes.append((str(o.get("name", "?")), str(o.get("severity", "?"))))
        halt_active = halt_active or bool(guard.get("halt_active", False))

    return DigestData(
        asof=asof,
        dry_run=dry_run,
        equity=equity,
        prev_equity=prev_equity,
        cash=cash,
        governance_live=sorted(governance_live or []),
        positions=sorted(positions),
        orders=orders,
        guard_worst_severity=guard_worst,
        guard_heartbeat=guard_heartbeat,
        guard_outcomes=guard_outcomes,
        halt_active=halt_active,
        jobs=_read_jobs(data_dir, asof),
    )


def render_facts(d: DigestData) -> str:
    """Deterministic plain-text fact sheet — the LLM input and the no-LLM fallback."""
    lines: list[str] = []
    mode = " (DRY-RUN — no live orders)" if d.dry_run else ""
    lines.append(f"Date: {d.asof.isoformat()}{mode}")

    if d.equity is not None:
        pl = d.day_pl
        pct = d.day_pl_pct
        pl_str = f" | day P&L {pl:+,.2f}" if pl is not None else ""
        pct_str = f" ({pct:+.2%})" if pct is not None else ""
        cash_str = f" | cash ${d.cash:,.2f}" if d.cash is not None else ""
        lines.append(f"Account: equity ${d.equity:,.2f}{pl_str}{pct_str}{cash_str}")
    else:
        lines.append("Account: no equity snapshot available")

    lines.append(
        "Governance live: " + (", ".join(d.governance_live) if d.governance_live else "(none)")
    )

    if d.positions:
        pos = ", ".join(f"{sym} {qty}" for sym, qty in d.positions)
        lines.append(f"Positions ({len(d.positions)}): {pos}")
    else:
        lines.append("Positions: flat (no open positions)")

    if d.orders:
        any_dry = any(o["dry_run"] for o in d.orders)
        tag = " (all DRY-RUN)" if all(o["dry_run"] for o in d.orders) else (" (some DRY-RUN)" if any_dry else "")
        order_str = "; ".join(
            f"{o['side'].upper()} {o['qty']} {o['symbol']}" for o in d.orders
        )
        lines.append(f"Orders today ({len(d.orders)}){tag}: {order_str}")
    else:
        lines.append("Orders today: none (no rebalance fired)")

    if d.guard_worst_severity is not None:
        gl = "; ".join(f"{n} {s}" for n, s in d.guard_outcomes) or "n/a"
        lines.append(f"Guardrails: {d.guard_worst_severity} — {gl}")

    if d.jobs:
        js = "; ".join(f"{j} {k} exit={e}" for j, k, e in d.jobs)
        lines.append(f"Scheduler: {js}")

    lines.append(f"Halt: {'ACTIVE — trading stopped' if d.halt_active else 'none'}")
    return "\n".join(lines)


def narrate(facts: str, *, settings: Any, client: Any | None = None) -> str | None:
    """Narrate the facts via Claude. Returns None when no key / on any error."""
    if client is None:
        if not getattr(settings, "anthropic_api_key", None):
            logger.info("analyst: no ANTHROPIC_API_KEY — using template-only digest")
            return None
        try:
            import anthropic
        except ImportError:
            logger.warning("analyst: anthropic SDK not installed — template-only digest")
            return None
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        resp = client.messages.create(
            model=getattr(settings, "anthropic_model", "claude-opus-4-8"),
            max_tokens=_MAX_TOKENS,
            system=[
                {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": facts}],
        )
    except Exception as exc:  # never let the daily job die on an API hiccup
        logger.error("analyst: Claude narration failed ({!r}) — template-only digest", exc)
        return None

    # Duck-typed extraction: works for the real SDK's TextBlock and the injected
    # test fake alike. getattr keeps mypy happy (the SDK content union doesn't
    # narrow through a getattr type-check).
    text = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    return text or None


def _slack_blocks(asof: date, body: str) -> list[dict[str, Any]]:
    return [
        {"type": "header", "text": {"type": "plain_text", "text": f"📊 quant daily digest — {asof.isoformat()}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": body[:2900]}},
    ]


def write_digest_artifact(
    artifact_dir: Path, asof: date, body: str, facts: str, used_llm: bool
) -> Path:
    """Write a committed markdown artifact (git history = the digest audit trail)."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{asof.isoformat()}.md"
    src = "Claude" if used_llm else "template (no LLM)"
    path.write_text(
        f"# Daily digest — {asof.isoformat()}\n\n{body}\n\n"
        f"---\n\n**Facts**\n\n```\n{facts}\n```\n\n_Source: {src}._\n",
        encoding="utf-8",
    )
    return path


def run_digest(
    *,
    data_dir: Path,
    asof: date,
    settings: Any,
    alerts: Any,
    artifact_dir: Path,
    client: Any | None = None,
    dry_run: bool = False,
    account: dict[str, float] | None = None,
    live_positions: list[tuple[str, int]] | None = None,
    governance_live: list[str] | None = None,
    halt_active: bool = False,
) -> DigestResult:
    """Build the digest, write the artifact, and deliver to Slack (unless dry_run)."""
    data = gather_digest_data(
        data_dir,
        asof,
        dry_run=dry_run,
        account=account,
        live_positions=live_positions,
        governance_live=governance_live,
        halt_active=halt_active,
    )
    facts = render_facts(data)
    narrative = narrate(facts, settings=settings, client=client)
    used_llm = narrative is not None
    body = narrative or facts

    artifact_path = write_digest_artifact(artifact_dir, asof, body, facts, used_llm)

    delivered = False
    if not dry_run:
        text = f"📊 quant daily digest — {asof.isoformat()}\n\n{body}"
        delivered = alerts.send_slack(text, blocks=_slack_blocks(asof, body))

    return DigestResult(
        body=body, facts=facts, used_llm=used_llm, delivered=delivered, artifact_path=artifact_path
    )
