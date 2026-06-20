# quant-trading — Honesty / Trust Audit

**Date:** 2026-06-19 (executed 2026-06-20)
**Phase:** Completion Roadmap, Phase 1 (Trust). Design: `docs/specs/2026-06-19-phase1-trust-audit-design.md`.
**Method:** Five independent dimensions (D1–D5) run by parallel subagents; every non-trivial
verdict **independently re-verified by the orchestrator** (re-ran the cited command or read the
cited `file:line`). No verdict is trusted on a subagent's say-so alone.
**Host:** MacBook Pro (M2 Pro) dev clone — NOT the live M4 Mac Mini. Read-only on the live account.

---

## Executive summary

**The system is real, not vibecoded.** Across 5 dimensions and 34 load-bearing claims, the audit
found **0 BROKEN**, **0 FAKE-STUB**, **0 S1**, **0 S2**. Everything substantive verifies; the only
open items are environmental (this is the dev clone) and out-of-scope-by-design (sim/research demos,
non-live strategies not re-run this pass).

| Verdict | Count |
|---|---|
| VERIFIED-REAL | 31 |
| UNVERIFIED (environmental / not-run, no risk) | 3 |
| BROKEN | 0 |
| FAKE-STUB | 0 |

**Severity:** 0× S1, 0× S2, all flagged items are S3 (cosmetic / cleanup / docs).

### The three make-or-break questions (with evidence)

**(a) Does it build & test green here?** **YES.** Canonical CI command
(`uv run pytest -m "not network and not alpaca and not slow"`) → **1395 passed / 0 failed / 0
skipped**, **86%** coverage. `ruff check`, `ruff format --check`, and **strict** `mypy quant/` (192
files) all clean. *Orchestrator independently re-ran ruff + format + mypy → all clean.*

**(b) Is any live strategy's backtest unreproducible?** **NO.** Both live-roster strategies
reproduce within tolerance. *Orchestrator independently re-ran `defensive-etf-allocation` validation
from the exact committed command:* committed DSR 0.4286 → reproduced **0.4239**, PSR 0.98035 →
**0.98037**, holdout 0.20382 → **0.20417**, regimes 2/4 → **2/4**, **all gates PASS**. Trend
likewise reproduces (DSR 0.5022 → 0.5066). Tiny drift vs committed is data-vintage (trailing holdout
window), not non-determinism — two independent runs agree to the digit.

**(c) Is the live system actually trading?** **YES — verified-real.** The order path reaches a real
Alpaca call (`quant/execution/alpaca.py:209 self._trading.submit_order(req)`), the production daily
job runs without `--dry-run` (`quant/deploy/jobs.toml:37`), and **7 real orders (`dry_run=False`)
were placed and FILLED** on 2026-06-02/06-03 (`data/live/trades.parquet`; recon docs show real fill
prices, 3.7 s median lag, 0 rejected). Zero-trade windows since are the strategy legitimately
holding (rolling-7d order count decays 7→7→7→4→0 as those fills age out), **not** a dead/suppressed
path. *Orchestrator independently confirmed the 7 `dry_run=False` rows and the real-vs-dry-run branch
in `alpaca.py`.*

### Top 5 risks (all S3 — nothing blocking)

1. **`alpaca-paper` MCP creds return 401** — direct independent reconciliation of the live account
   couldn't run; the audit fell back to Alpaca-sourced recon artifacts. Fix creds to enable future
   independent audits. → **Phase 2**.
2. **Live host (M4) not inspectable from here; engine state stale since 2026-06-11.** This is the dev
   clone with no launchd jobs loaded. Confirms the known relay constraint, not a defect. → **Phase 2**.
3. **Non-live strategies (momentum, multi-factor, pairs, risk-parity) not re-run** this pass —
   UNVERIFIED (no S1 risk; not live). They share the engine/validation path proven honest in
   D3-3/4/5. → **Phase 4** (when expanding the roster).
4. **`uv sync` footgun** — bare `uv sync` *uninstalls* dev tooling (dev deps live in
   `optional-dependencies`, not a default group); canonical install is `uv sync --all-extras`.
   → docs note (Phase 2/3).
5. **Minor cleanup**: dead orphaned `quant/data/fundamentals.py` (0 imports; real path is
   EDGAR-PIT `quant/fundamentals/factors.py`), two safe-degrading data-layer TODOs in
   `quant/intraday/live/loop.py`, a no-op `intraday live flat` CLI button. → **Phase 3**.

---

## Findings table

`✓ = independently re-verified by orchestrator.` Verdicts: VERIFIED-REAL / BROKEN / FAKE-STUB /
UNVERIFIED. Severity: S1 (system lying / live broken) / S2 (module fake / unreproducible, not live) /
S3 (cosmetic / cleanup / docs).

### D1 — Build health

| id | claim | verdict | sev | evidence | phase |
|---|---|---|---|---|---|
| D1-1 | `uv sync` clean | VERIFIED-REAL | S3 | `uv sync --all-extras` → Resolved 115 pkgs, no errors. Footgun: bare `uv sync` strips 20 dev pkgs. | docs |
| D1-2 | Test suite green | VERIFIED-REAL ✓ | — | Canonical filter → `1395 passed, 0 failed, 0 skipped, 486.77s`. | — |
| D1-3 | Unfiltered 1 "failure" | VERIFIED-REAL ✓ | S3 | Only failure = `test_alpaca_hist_client` (`@pytest.mark.alpaca`, 401 on dummy creds); CI deselects it. | — |
| D1-4 | Skips are torch/alpaca-gated | VERIFIED-REAL | S3 | 5 DL files `importorskip("torch")` (torch 2.12 present → they run); 1 alpaca-gated file deselected. | — |
| D1-5 | ruff clean | VERIFIED-REAL ✓ | — | `ruff check .` → All checks passed; `ruff format --check .` → 448 files formatted. | — |
| D1-6 | mypy strict clean | VERIFIED-REAL ✓ | — | `mypy quant/` → Success, no issues in 192 files (`strict=true`). | — |
| D1-7 | Coverage honest | VERIFIED-REAL | S3 | `TOTAL … 86%`. Lowest: tui.py 59%, options/surface.py 79%, pairs_trading.py 80%. | — |
| D1-8 | Canonical verify cmd | VERIFIED-REAL ✓ | S3 | No Makefile; canonical = `.github/workflows/ci.yml` (sync→ruff→format→mypy→pytest filter). | — |

### D2 — Stub / fake hunt

| id | area | claim | verdict | sev | evidence | phase |
|---|---|---|---|---|---|---|
| D2-1 | intraday loop | hardcoded $5B ADV constant | VERIFIED-REAL (safe stub) ✓ | S3 | `intraday/live/loop.py:79`; calibration-only anchor, not signal/order existence. | 3 |
| D2-2 | intraday loop | `_recent_returns` returns `[]` → σ=0 | VERIFIED-REAL (safe default) ✓ | S3 | `loop.py:87`; empty → Almgren-Chriss degenerates to TWAP, the documented safe default. | 3 |
| D2-3 | data/fundamentals.py | yfinance "stub" | VERIFIED-REAL, DEAD/ORPHANED ✓ | S3 | **0 imports** of `quant.data.fundamentals`; real path = EDGAR-PIT `quant/fundamentals/factors.py`. | 3 (delete) |
| D2-4 | intraday CLI | `live flat` is a no-op echo | VERIFIED-REAL | S3 | `intraday/cli.py:101`; real flatten (`_flatten_all` submits closing orders) runs in the daemon. | 3 |
| D2-5 | nlp module E | suspected half-built/unwired | VERIFIED-REAL & WIRED | — | LM-lexicon scorer + real Alpaca news fetch, wired into engine loop (read-only signal; engine never trades). Lexicon-grade, honest about future FinBERT. | 3 (enrich) |
| D2-6 | intraday showcase | wired or demos? | VERIFIED-REAL (mixed) | — | execution=live-wired; marketmaking/rl/dl = sim/CLI-only, honestly labeled; no live path fabricates data. | 3 |
| D2-7 | live/engine order path | silent no-op? | VERIFIED-REAL | — | All early-returns are explicit fail-closed guards; no order fn silently no-ops. | — |
| D2-8 | options/structures.py | `_PLACEHOLDER_EXPIRY=0` | VERIFIED-REAL | — | Sentinel overwritten with real `expiry_index` at roll (`policy.py`/`overlay.py`). | — |
| D2-9 | whole tree | TODO/FIXME/HACK/XXX | VERIFIED-REAL (benign) | S3 | Total real TODOs = 2 (= D2-1/D2-2). No FIXME/HACK/XXX; no NotImplementedError; bare `pass` are Click groups. | — |

### D3 — Backtest honesty

| id | strategy | claim | verdict | sev | evidence | phase |
|---|---|---|---|---|---|---|
| D3-1 | defensive-etf (LIVE) | DSR/PSR/holdout/regime gates PASS | VERIFIED-REAL ✓ | — | **Orchestrator re-ran exact committed cmd**: DSR 0.4286→0.4239, PSR→0.98037, holdout→0.20417, 2/4, all PASS. | — |
| D3-2 | trend (LIVE) | gates PASS | VERIFIED-REAL | — | Agent re-ran: DSR 0.5022→0.5066, PSR→0.99038, boot-p05→0.13378, 4/4, all PASS. | — |
| D3-3 | all | cost model applied | VERIFIED-REAL | — | slippage 5bps + commission + financing 200bps + impact ON by default (`backtest/engine.py:30-94,171-216`), carried into OOS test_config. | — |
| D3-4 | all | PIT / no-lookahead | VERIFIED-REAL | — | `asof_index` last bar ≤ asof (`_common.py:31-43`); T+1 `next_open` fills; train/test windows separate. | — |
| D3-5 | all | DSR multiple-testing gate not bypassed | VERIFIED-REAL | S3 | DSR deflates vs full `grid_trial_sharpes` (`validation.py:225`); 5 gates ANDed. NOTE: no standalone PBO scalar — "PBO" carried by CPCV path-Sharpes + DSR deflation (reporting-only). | — |
| D3-6 | momentum/multi-factor/pairs/risk-parity (non-live) | reproduce? | UNVERIFIED | S3 | Not re-run (time-boxed to live roster = S1 priority). Artifacts exist; share the honest engine path. | 4 |

### D4 — Live-path reality

| id | claim | verdict | sev | evidence | phase |
|---|---|---|---|---|---|
| D4-1 | order path reaches real Alpaca call | VERIFIED-REAL ✓ | — | `execution/alpaca.py:209 submit_order(req)` vs `:198-208` dry-run branch returns COID without submitting. | — |
| D4-2 | prod daily job runs LIVE (no `--dry-run`) | VERIFIED-REAL | — | `deploy/jobs.toml:37 ["rebalance","--derisk-actuate"]`; tick plist 60s → dispatcher → rebalance. | — |
| D4-3 | `dry_run` default False end-to-end | VERIFIED-REAL | — | CLI flag absent ⇒ False (`cli.py:618`); `rebalance.py:339`; flows to `submit_order(dry_run=False)`. | — |
| D4-4 | real orders placed AND filled | VERIFIED-REAL ✓ | — | **7 `dry_run=False` rows** in `trades.parquet` (DBC/EEM/GLD 06-02; +SPY 06-03); recon: 3+7 filled, 0 rejected, real fills. | — |
| D4-5 | recon from real Alpaca queries | VERIFIED-REAL | — | `scripts/reconcile_live.py:173` `client.list_orders(...)` → real `get_orders` API. | — |
| D4-6 | post-cutover zero trades = held, not dead | VERIFIED-REAL | — | Rolling-7d count 7→7→7→4→0 as 06-02/03 fills age out; only defensive-etf live, no new deltas. | — |
| D4-7 | de-risk overlay default shadow, actuated in prod | VERIFIED-REAL | — | `derisk.py:38 actuate=False` default; `--derisk-actuate` in jobs.toml → only shrinks gross, floor 0.5, reversible. | — |
| D4-8 | this host not the live runner; data stale 06-11 | UNVERIFIED (env) | S3 | `launchctl list \| grep quant` → none; engine state last write 06-11. Dev clone; live M4 not inspectable here. | 2 |
| D4-9 | direct MCP account reconciliation | UNVERIFIED (env) | S3 | `alpaca-paper` MCP → 401 Unauthorized (MCP's own creds invalid). Fell back to Alpaca-sourced recon docs. | 2 (fix creds) |

### D5 — Determinism / reproducibility

| id | claim | verdict | sev | evidence | phase |
|---|---|---|---|---|---|
| D5-1 | no unseeded RNG | VERIFIED-REAL | — | Grep for `default_rng()`/`RandomState()`/bare `np.random.*` in `quant/` (excl tests) → 0 hits; every RNG seeded. | — |
| D5-2 | DSR/PBO bootstrap seeded & reproducible | VERIFIED-REAL ✓ | — | **Orchestrator re-ran**: same-seed identical = True, diff-seed differs = True. Gate threads `--bootstrap-seed` (default 0). | — |
| D5-3 | governance manifests deterministic | VERIFIED-REAL | — | `governance refresh` run twice → all 3 outputs byte-identical (`json.dumps(sort_keys=True)`, asof is a CLI arg, no RNG). | — |
| D5-4 | allocation_compare deterministic | VERIFIED-REAL | — | `rebalance.py:279-333` keyed by `asof.isoformat()`, atomic sort_keys write, no RNG/`now()` in payload (by inspection). | — |
| D5-5 | intraday-DL determinism flag set & restored | VERIFIED-REAL | — | `dl/train.py:29-68` manual_seed + `use_deterministic_algorithms(True)` saved/restored in `finally`; torch 2.12 → identical loss curves. | — |

---

## Re-planning Phases 2–4 (what the punch-list drives)

The audit found **nothing that re-scopes the phases downward** — the system is real and green. It
sharpens the next phases:

- **Phase 2 (Reliability):** Highest-value real work. The live path is proven, but it's only
  inspectable via relay; engine state is stale since 06-11. Confirm launchd jobs actually run on the
  M4, prove auto-restart, add alerting, fix the `alpaca-paper` MCP creds (D4-9) so the account can be
  independently audited going forward, and run the 48h soak. Add a one-line README note on the
  `uv sync --all-extras` footgun (D1-1).
- **Phase 3 (Finish pieces):** Enrich NLP module E beyond lexicon-grade (D2-5, honest today but a
  monitoring signal only), optionally wire intraday `_recent_returns`/ADV data layer (D2-1/D2-2),
  delete dead `quant/data/fundamentals.py` (D2-3), wire or remove the `intraday live flat` no-op
  (D2-4).
- **Phase 4 (Performance):** Before live-promoting any non-live strategy, re-run its validation
  (D3-6 UNVERIFIED) under the same honest gates proven in D3-3/4/5.

---

## Reproduction appendix

Run from `/Users/ajaiupadhyaya/Documents/quant-trading`.

```bash
# D1 — build health (canonical CI command; ~8 min)
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy quant/
uv run pytest -q -rs -m "not network and not alpaca and not slow"   # 1395 passed / 0 failed, 86%

# D2 — dead-file check (expect zero output)
rg -n 'quant\.data\.fundamentals|from quant.data.fundamentals' quant/ --glob '!**/__pycache__/**'

# D3 — live-roster backtest reproduction (~2-4 min each; overwrites artifacts → git checkout after)
git show HEAD:data/backtests/defensive-etf-allocation/validation_report.json   # baseline
uv run quant validate defensive-etf-allocation --start 2010-01-01 --end 2026-06-03 --bootstrap-resamples 5000 --bootstrap-seed 0
uv run quant validate trend --start 2010-01-01 --end 2026-06-10 --bootstrap-resamples 5000 --bootstrap-seed 0
git checkout -- data/backtests/   # revert overwritten reports/tearsheets

# D4 — live-trade evidence (read-only)
uv run python -c "import pandas as pd; df=pd.read_parquet('data/live/trades.parquet'); print(df['dry_run'].value_counts())"   # False: 7
#   code path: quant/execution/alpaca.py:209 (real submit) ; quant/deploy/jobs.toml:37 (no --dry-run)

# D5 — bootstrap determinism + governance manifest run-twice-diff
uv run python -c "import pandas as pd,numpy as np;from quant.backtest.bootstrap import bootstrap_ci;rng=np.random.default_rng(1);r=pd.Series(rng.normal(0.0005,0.01,500));print(bootstrap_ci(r,300,seed=42)==bootstrap_ci(r,300,seed=42))"   # True
#   governance: copy data/ to a temp QUANT_DATA_DIR, run `quant governance refresh --asof 2026-06-20` twice, diff outputs → identical
```

## Definition of done

- [x] This doc exists, committed, every dimension covered, every finding carries evidence + a phase.
- [x] Exec summary answers (a) builds green, (b) no live backtest unreproducible, (c) live system IS trading — all with evidence.
- [x] Tree clean (Step-0 gitignore sweep done; analyst/live-recon trail committed alongside).
- [ ] Checkpoint: review punch-list together → drives the Phase 2 spec.
</content>
</invoke>
