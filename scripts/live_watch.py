"""Stdout event emitter for live paper-trading monitoring.

Polls Alpaca every ~60s and writes a compact one-line event to stdout when
any of these change materially:

* Account equity changed by more than $EQUITY_DELTA (default $25)
* A position was added, removed, or its qty changed
* Day P&L moved by more than $PNL_DELTA (default $50)

Intended to be wrapped by the Claude Monitor tool — each stdout line becomes
a notification. The script never exits on its own; kill it with SIGINT or
let the Monitor timeout do it.

Usage::

    python scripts/live_watch.py            # default 60s poll
    POLL_S=30 python scripts/live_watch.py  # 30s poll
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime

from quant.execution.alpaca import AlpacaClient
from quant.util.config import Settings


def _snapshot(client: AlpacaClient) -> dict[str, object]:
    acct = client.account()
    positions = client.positions()
    by_sym = {p.symbol: int(p.qty) for p in positions}
    total_unrealized = sum(float(p.unrealized_pl) for p in positions)
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "positions": by_sym,
        "unrealized_pl": total_unrealized,
        "n_positions": len(positions),
    }


def _diff_positions(prev: dict[str, int], cur: dict[str, int]) -> list[str]:
    """Return human-readable diff strings for position changes."""
    diffs: list[str] = []
    for sym in sorted(set(prev) | set(cur)):
        p, c = prev.get(sym, 0), cur.get(sym, 0)
        if p == c:
            continue
        if p == 0:
            diffs.append(f"+{sym}({c})")
        elif c == 0:
            diffs.append(f"-{sym}({p})")
        else:
            diffs.append(f"{sym}:{p}->{c}")
    return diffs


def main() -> int:
    poll_s = int(os.environ.get("POLL_S", "60"))
    eq_delta = float(os.environ.get("EQUITY_DELTA", "25"))
    pnl_delta = float(os.environ.get("PNL_DELTA", "50"))

    settings = Settings()  # type: ignore[call-arg]
    client = AlpacaClient(settings=settings)

    prev: dict[str, object] | None = None
    sys.stdout.write(
        f"watch armed poll={poll_s}s equity_delta=${eq_delta:.0f} pnl_delta=${pnl_delta:.0f}\n"
    )
    sys.stdout.flush()

    while True:
        try:
            cur = _snapshot(client)
        except Exception as exc:  # broad: don't crash the watch on transient API blips
            sys.stdout.write(f"[{datetime.now(UTC):%H:%M:%SZ}] poll-error: {exc}\n")
            sys.stdout.flush()
            time.sleep(poll_s)
            continue

        emit = False
        reasons: list[str] = []

        if prev is None:
            emit = True
            reasons.append("initial")
        else:
            de = float(cur["equity"]) - float(prev["equity"])
            dp = float(cur["unrealized_pl"]) - float(prev["unrealized_pl"])
            pos_diff = _diff_positions(prev["positions"], cur["positions"])  # type: ignore[arg-type]
            if abs(de) >= eq_delta:
                emit = True
                reasons.append(f"Δeq=${de:+.0f}")
            if abs(dp) >= pnl_delta:
                emit = True
                reasons.append(f"ΔuPL=${dp:+.0f}")
            if pos_diff:
                emit = True
                reasons.append("pos:" + ",".join(pos_diff[:8]))

        if emit:
            ts = datetime.now(UTC).strftime("%H:%M:%SZ")
            line = (
                f"[{ts}] eq=${cur['equity']:,.0f} uPL=${cur['unrealized_pl']:+,.0f} "
                f"cash=${cur['cash']:,.0f} pos={cur['n_positions']} | " + " ".join(reasons)
            )
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

        prev = cur
        time.sleep(poll_s)


if __name__ == "__main__":
    sys.exit(main())
