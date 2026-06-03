# Resume here (cross-machine handoff) — 2026-06-02

Everything is committed + pushed to **`feat/m4-deploy-e1`** on
`https://github.com/ajaiupadhyaya/quant-trading.git`. This note is the fastest way
to get back up to speed on another computer (or in a fresh Claude session).

## Current state (one paragraph)
The system is **LIVE on Alpaca paper** as of 2026-06-02. The M4 Mac mini is the
sole 24/7 host (launchd tick + guard daemons, running from its working tree). The
first autonomous paper rebalance filled at 15:55 ET (`DBC/EEM/GLD`, ~$1.0M book).
Only `defensive-etf-allocation` is governance-LIVE; the other 5 strategies are
quarantined by the evidence battery. A read-only Claude decision-maker (Phase A
brief + Phase B advise-and-log) and a portfolio-risk view (VaR/CVaR/vol/beta/
asset-class) are wired in. Branch is **not** merged to `main`.

## Set up the OTHER computer (development only — see warning)
```bash
git clone https://github.com/ajaiupadhyaya/quant-trading.git
cd quant-trading
git checkout feat/m4-deploy-e1
cp .env.example .env      # then fill in real secrets (see below)
uv sync --all-extras
uv run quant doctor       # expect 7/7
```
`.env` is gitignored (secrets never leave a machine). Fill in: `ALPACA_API_KEY`,
`ALPACA_SECRET_KEY` (paper), `FRED_API_KEY`, `SLACK_WEBHOOK_URL`,
`ANTHROPIC_API_KEY`. Everything else in `.env.example` can stay default/blank.

### ⚠️ Do NOT run the launchd host on a second machine
The M4 is the live host. Running `deploy/install.sh` (the launchd agents) on a
second computer would have TWO hosts trading the SAME paper account → conflicting/
double orders. The other computer is for **development only**: edit, test, commit,
push. To deploy changes to the live host, `git pull` on the **M4** (it does not
auto-pull). The M4 keeps trading from its current tree until you pull.

## What travels with the repo vs. what doesn't
- **In git (travels):** all code, governance state (`data/governance/`), the live
  book (`data/live/`), the roadmap, day-1 report artifacts.
- **NOT in git (recreate):** `.env` (secrets — you have them); `data/raw/` bar
  caches (regenerate with `uv run quant data refresh`); Claude Code's local memory.

## Key docs / pointers
- **Forward plan:** `docs/specs/2026-06-02-raise-the-ceiling-roadmap.md` — the
  sequenced roadmap + the **do-not-do-yet** (human-gated) list. Read this first.
- **Operator runbook:** `deploy/README.md`. **Deploy design:**
  `docs/superpowers/specs/2026-06-02-m4-deployment-e1-design.md`.
- **What shipped 2026-06-02:** `git log --oneline a84bd45..HEAD` (safety fixes →
  cutover → decision-maker A/B → portfolio risk → pairs fixes → roadmap).

## New commands added 2026-06-02
- `uv run quant analyst brief [--dry-run]` — Claude Phase A context brief (read-only)
- `uv run quant analyst propose` — Claude Phase B advisory proposals (clamped, logged, applies nothing)
- `uv run quant risk portfolio` — VaR/CVaR/vol/beta/asset-class of the live book

## ⚠️ The one trap to remember
Do NOT autonomously "fix" the DSR trial count (`validation.py:206`): the corrected
count drops `defensive-etf` DSR ~0.602 → ~0.246 (below the 0.30 gate) and would
quarantine the only live strategy at the weekly governance refresh. See Phase 0 of
the roadmap for the safe sequence.

## Next decisions (when you're ready)
1. Review the roadmap; decide on the DSR question (Phase 0).
2. Optionally start Phase 1 (pure foundations — safe) of the roadmap.
3. Decide whether to merge `feat/m4-deploy-e1` → `main`.
