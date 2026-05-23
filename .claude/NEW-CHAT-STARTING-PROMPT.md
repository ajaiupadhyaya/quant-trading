# Starting prompt for the implementation chat

Paste the following message into a NEW Claude Code chat after you `cd ~/Documents/quant-trading`:

---

I'm starting implementation of the `quant-trading` project. The full design spec is at `docs/specs/2026-05-23-quant-trading-design.md` — please read it now, then we'll go.

Some context the spec doesn't repeat:
- I have a live Alpaca paper account; credentials live as env vars (`ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `ALPACA_PAPER=true`) — we'll wire them in via GitHub secrets later.
- This is a NEW repo. Git is initialized but no implementation code exists yet, only the spec + README. Build everything inside `quant/`.
- A reference implementation of cross-sectional momentum, multi-factor, pairs trading, plus the walk-forward engine + vectorbt wrapper, exists in `~/Documents/news-dashboard/backend/app/quant/`. Feel free to port from there — but the new versions should incorporate the SOTA enhancements per §2 of the spec (residual momentum, Kalman hedge ratios, etc.). Don't just copy v1; refine.
- We agreed on a 6-week milestone breakdown in §7.1 of the spec. Use that as the implementation cycle.

Next step: invoke the `superpowers:writing-plans` skill to produce the implementation plan for week 1 (repo skeleton + CLI + data layer). The skill will read the spec and propose a detailed task breakdown. Then we execute via subagent-driven development.

---

That's it. Once you paste that, the new chat picks up exactly where this brainstorm left off.
