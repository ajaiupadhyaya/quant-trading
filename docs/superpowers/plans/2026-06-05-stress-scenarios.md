# Scenario / Stress-Shock Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure, WARN-only/read-only scenario-stress engine (historical replays + hypothetical asset-class/symbol shocks) to the live book, surfaced via the Guard-5 artifact, a `quant risk scenarios` CLI, and the analyst brief.

**Architecture:** A standalone pure module `quant/risk/scenarios.py` mirrors `portfolio.py` (frozen dataclasses + pure `compute_stress` + best-effort `live_stress` + `render`). The stress→violation mapping is threaded through the existing pure `build_portfolio_risk_gate` (new optional duck-typed `stress` kwarg) so it stays unit-testable and never mutates frozen results. Guard 5 computes `live_stress`, passes it into the gate, and folds it into the same per-run artifact. CLI + brief reuse the established `risk_portfolio` / `_portfolio_risk` patterns. Read-only and fail-open throughout — cannot touch `netted`.

**Tech Stack:** Python 3.12, numpy, pandas, Click (CLI), pytest, `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-06-05-stress-scenarios-design.md`

**Working directory:** the `.worktrees/stress-scenarios` worktree on branch `feat/stress-scenarios` (the main checkout is the live launchd host's tree — do NOT switch it off `main`). Run all commands from the worktree root.

---

## File Structure

- **Create** `quant/risk/scenarios.py` — pure stress engine: `HistoricalScenario`, `HypotheticalScenario`, `ScenarioResult`, `StressReport`, `default_scenarios()`, `compute_stress()`, `live_stress()`, `StressReport.render()`.
- **Create** `tests/risk/test_scenarios.py` — pure unit tests for the engine + gate violation mapping.
- **Modify** `quant/risk/portfolio.py` — add `PortfolioRiskLimits.max_scenario_loss`, add `stress` field to `RiskGateResult`, add `stress` kwarg + stress-violation logic to `build_portfolio_risk_gate`.
- **Modify** `quant/live/rebalance.py` — Guard 5: compute `live_stress`, pass to `build_portfolio_risk_gate`, extend `_write_portfolio_risk_gate_artifact` with a `"stress"` key.
- **Modify** `tests/live/test_rebalance.py` — regression: stress CheckResult ok, `netted` byte-identical in WARN, artifact has `stress`.
- **Modify** `quant/cli.py` — new `@risk.command("scenarios")`.
- **Modify** `tests/test_cli.py` (or the existing risk-CLI test module) — smoke test for `quant risk scenarios`.
- **Modify** `quant/analyst/context.py` — `AnalystContext.stress` field, `_stress()` helper, builder wiring, brief render line.
- **Modify** `tests/analyst/test_context.py` (or equivalent) — brief includes a stress line when stress present.

Conventions for shock keys: **uppercase = symbol** (`"TLT"`), **lowercase = asset-class bucket** (`"bond"`, matching `_SECTOR_MAP` values). Per held symbol, a symbol-level shock overrides its class-level shock. Sign convention: a `ScenarioResult.pnl_pct` is signed (negative = loss); `StressReport.worst_loss` = `-(min pnl_pct over computable results)` so positive = loss (parallels `var_95`).

---

### Task 1: Pure stress engine — dataclasses + `compute_stress`

**Files:**
- Create: `quant/risk/scenarios.py`
- Test: `tests/risk/test_scenarios.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/risk/test_scenarios.py`:

```python
"""Pure unit tests for the scenario/stress engine (no data/network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.risk.scenarios import (
    HistoricalScenario,
    HypotheticalScenario,
    ScenarioResult,
    StressReport,
    compute_stress,
    default_scenarios,
)


def _panel(dates, data):
    return pd.DataFrame(data, index=pd.to_datetime(dates))


def test_hypothetical_class_level_shock():
    # 50% SPY (equity), 50% TLT (bond); equity -20%, bond +5%
    weights = {"SPY": 0.5, "TLT": 0.5}
    scen = HypotheticalScenario("eq-crash", {"equity": -0.20, "bond": 0.05})
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=[scen])
    res = rep.results[0]
    assert res.computable
    assert res.pnl_pct == 0.5 * -0.20 + 0.5 * 0.05  # = -0.075
    assert res.by_class["equity"] == 0.5 * -0.20
    assert res.by_class["bond"] == 0.5 * 0.05


def test_symbol_override_beats_class():
    # rate shock: bond class -7% but TLT overridden to -15%
    weights = {"TLT": 1.0}
    scen = HypotheticalScenario("rate", {"bond": -0.07, "TLT": -0.15})
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=[scen])
    assert rep.results[0].pnl_pct == -0.15


def test_unshocked_asset_contributes_zero():
    weights = {"SPY": 0.5, "GLD": 0.5}
    scen = HypotheticalScenario("eq-only", {"equity": -0.10})  # no gold shock
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=[scen])
    res = rep.results[0]
    assert res.pnl_pct == 0.5 * -0.10  # GLD contributes 0
    assert "GLD" in res.missing_symbols


def test_historical_replay_cumulative_return():
    # SPY daily returns over the window: +10% then -20% => cumret = 1.1*0.8-1 = -0.12
    dates = ["2008-09-01", "2008-09-02", "2008-09-03"]
    returns = _panel(dates, {"SPY": [0.0, 0.10, -0.20]})
    weights = {"SPY": 1.0}
    scen = HistoricalScenario("gfc", start=pd.Timestamp("2008-09-02").date(),
                              end=pd.Timestamp("2008-09-03").date())
    rep = compute_stress(weights, returns=returns, scenarios=[scen])
    res = rep.results[0]
    assert res.computable
    assert abs(res.pnl_pct - (1.10 * 0.80 - 1.0)) < 1e-12  # -0.12


def test_historical_missing_symbol_degrades_but_still_computes():
    dates = ["2008-09-02", "2008-09-03"]
    returns = _panel(dates, {"SPY": [0.10, -0.20]})  # no TLT column
    weights = {"SPY": 0.5, "TLT": 0.5}
    scen = HistoricalScenario("gfc", start=pd.Timestamp("2008-09-02").date(),
                              end=pd.Timestamp("2008-09-03").date())
    rep = compute_stress(weights, returns=returns, scenarios=[scen])
    res = rep.results[0]
    assert res.computable  # SPY computed
    assert "TLT" in res.missing_symbols
    assert abs(res.pnl_pct - 0.5 * (1.10 * 0.80 - 1.0)) < 1e-12  # only SPY half


def test_worst_loss_and_scenario_selection():
    weights = {"SPY": 1.0}
    scens = [
        HypotheticalScenario("mild", {"equity": -0.05}),
        HypotheticalScenario("severe", {"equity": -0.30}),
        HypotheticalScenario("gain", {"equity": 0.10}),
    ]
    rep = compute_stress(weights, returns=pd.DataFrame(), scenarios=scens)
    assert rep.worst_scenario == "severe"
    assert abs(rep.worst_loss - 0.30) < 1e-12  # positive = loss
    assert rep.computable


def test_all_gains_worst_loss_is_negative():
    weights = {"SPY": 1.0}
    rep = compute_stress(weights, returns=pd.DataFrame(),
                         scenarios=[HypotheticalScenario("g", {"equity": 0.10})])
    assert rep.worst_loss < 0  # best-case-only => "loss" is negative


def test_empty_weights_not_computable():
    rep = compute_stress({}, returns=pd.DataFrame(),
                        scenarios=[HypotheticalScenario("x", {"equity": -0.2})])
    assert not rep.computable
    assert rep.worst_loss is None
    assert rep.worst_scenario is None


def test_default_scenarios_library_shape():
    scens = default_scenarios()
    names = {s.name for s in scens}
    assert {"2008-GFC", "2020-COVID", "2022-rate-selloff", "2013-taper-tantrum"} <= names
    assert {"equity-crash-20", "rate-shock-+100bp", "stagflation", "risk-off-flight"} <= names
    # historical have date windows, hypothetical have shock dicts
    for s in scens:
        if isinstance(s, HistoricalScenario):
            assert s.start < s.end
        else:
            assert isinstance(s, HypotheticalScenario) and s.shocks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/risk/test_scenarios.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.risk.scenarios'`.

- [ ] **Step 3: Write the module**

Create `quant/risk/scenarios.py`:

```python
"""Scenario / stress-shock evaluation for the live book (Raise-the-Ceiling Phase 2).

How much would today's holdings lose under a 2008-style crash, a +100bp rate
shock, etc.? Both scenario kinds reduce to the same kernel — apply a per-asset
return shock to current signed weights -> portfolio P&L%:

    pnl_pct = sum_i  weight_i * shock_i

This is a standalone, READ-ONLY analysis layer (mirrors ``portfolio.py``): it is
WARN-only, never wired to block an order, and fail-open at every live entry point.
``compute_stress`` is pure (weights + returns/shocks) so it is trivially testable.

Shock-key convention: UPPERCASE keys are symbols (``"TLT"``), lowercase keys are
asset-class buckets (``"bond"``, matching ``portfolio._SECTOR_MAP`` values). A
symbol-level shock overrides its class-level shock for that symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.risk.portfolio import _SECTOR_MAP, weights_from_positions
from quant.util.logging import logger


@dataclass(frozen=True)
class HistoricalScenario:
    """Replay each held asset's OWN cumulative simple return over [start, end]."""

    name: str
    start: date
    end: date
    description: str = ""
    kind: str = "historical"


@dataclass(frozen=True)
class HypotheticalScenario:
    """Apply a shock vector (asset-class bucket OR symbol -> return shock)."""

    name: str
    shocks: dict[str, float]
    description: str = ""
    kind: str = "hypothetical"


Scenario = HistoricalScenario | HypotheticalScenario


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    kind: str
    pnl_pct: float | None  # signed; negative = loss; None if nothing computable
    by_class: dict[str, float] = field(default_factory=dict)  # per-class P&L contribution
    missing_symbols: tuple[str, ...] = ()  # held names with no shock/no data
    computable: bool = True


@dataclass(frozen=True)
class StressReport:
    results: tuple[ScenarioResult, ...]
    worst_loss: float | None  # -(min pnl over computable); positive = loss
    worst_scenario: str | None
    computable: bool  # at least one scenario computable
    degraded: tuple[str, ...] = ()  # scenario names that were not computable

    def render(self) -> str:
        """Compact one-block summary for CLI/brief/Slack."""
        if not self.computable or self.worst_loss is None:
            return "stress: n/a (no computable scenarios)"
        head = f"worst {self.worst_scenario} {self.worst_loss:+.1%} loss"
        parts = []
        for r in self.results:
            if r.pnl_pct is None:
                parts.append(f"{r.name} n/a")
            else:
                parts.append(f"{r.name} {r.pnl_pct:+.1%}")
        return head + " | " + ", ".join(parts)


def _historical_shock(sym: str, returns: pd.DataFrame, start: date, end: date) -> float | None:
    """Cumulative simple return of ``sym`` over [start, end], or None if no data."""
    if returns is None or returns.empty or sym not in returns.columns:
        return None
    col = returns[sym]
    mask = (col.index >= pd.Timestamp(start)) & (col.index <= pd.Timestamp(end))
    window = col[mask].dropna()
    if window.empty:
        return None
    return float(np.prod(1.0 + window.to_numpy()) - 1.0)


def _hypothetical_shock(sym: str, shocks: dict[str, float]) -> float | None:
    """Symbol-level shock overrides class-level; None if neither present."""
    if sym.upper() in shocks:
        return float(shocks[sym.upper()])
    bucket = _SECTOR_MAP.get(sym.upper(), "other")
    if bucket in shocks:
        return float(shocks[bucket])
    return None


def _evaluate(weights: dict[str, float], shock_for_symbol) -> ScenarioResult | dict:
    """Apply ``shock_for_symbol(sym) -> float | None`` across weights.

    Returns a dict of the raw pieces; the caller wraps it into a ScenarioResult
    (so name/kind are added once). An asset whose shock is None contributes 0 and
    is recorded in missing_symbols.
    """
    pnl = 0.0
    by_class: dict[str, float] = {}
    missing: list[str] = []
    any_shocked = False
    for sym, w in weights.items():
        s = shock_for_symbol(sym)
        if s is None:
            missing.append(sym)
            s = 0.0
        else:
            any_shocked = True
        contrib = float(w) * s
        pnl += contrib
        bucket = _SECTOR_MAP.get(sym.upper(), "other")
        by_class[bucket] = by_class.get(bucket, 0.0) + contrib
    return {
        "pnl_pct": pnl if any_shocked else None,
        "by_class": {k: v for k, v in sorted(by_class.items(), key=lambda kv: kv[1])},
        "missing_symbols": tuple(sorted(missing)),
        "computable": any_shocked,
    }


def compute_stress(
    weights: dict[str, float],
    returns: pd.DataFrame,
    scenarios,
) -> StressReport:
    """Pure: evaluate ``scenarios`` against ``weights`` (+ ``returns`` for historical)."""
    nonzero = {k: float(v) for k, v in weights.items() if abs(float(v)) > 0}
    results: list[ScenarioResult] = []
    degraded: list[str] = []
    for scen in scenarios:
        if isinstance(scen, HistoricalScenario):
            pieces = _evaluate(
                nonzero, lambda sym, s=scen: _historical_shock(sym, returns, s.start, s.end)
            )
        else:
            pieces = _evaluate(nonzero, lambda sym, s=scen: _hypothetical_shock(sym, s.shocks))
        res = ScenarioResult(name=scen.name, kind=scen.kind, **pieces)
        results.append(res)
        if not res.computable:
            degraded.append(scen.name)

    computable_pnls = [(r.name, r.pnl_pct) for r in results if r.pnl_pct is not None]
    if not computable_pnls:
        return StressReport(
            results=tuple(results),
            worst_loss=None,
            worst_scenario=None,
            computable=False,
            degraded=tuple(degraded),
        )
    worst_name, worst_pnl = min(computable_pnls, key=lambda kv: kv[1])
    return StressReport(
        results=tuple(results),
        worst_loss=-worst_pnl,
        worst_scenario=worst_name,
        computable=True,
        degraded=tuple(degraded),
    )


def default_scenarios() -> tuple[Scenario, ...]:
    """Curated stress library (see spec). Windows are peak->trough of the episode."""
    return (
        HistoricalScenario("2008-GFC", date(2008, 9, 1), date(2009, 3, 9),
                           "Lehman to the market bottom"),
        HistoricalScenario("2020-COVID", date(2020, 2, 19), date(2020, 3, 23),
                           "COVID crash"),
        HistoricalScenario("2022-rate-selloff", date(2022, 1, 1), date(2022, 10, 14),
                           "2022 rate-driven 60/40 drawdown"),
        HistoricalScenario("2013-taper-tantrum", date(2013, 5, 22), date(2013, 6, 24),
                           "Taper tantrum"),
        HypotheticalScenario("equity-crash-20",
                            {"equity": -0.20, "real_estate": -0.25, "commodity": -0.10,
                             "gold": 0.05, "bond": 0.05}, "Broad equity crash, flight to quality"),
        HypotheticalScenario("rate-shock-+100bp",
                            {"TLT": -0.15, "IEF": -0.07, "equity": -0.05,
                             "real_estate": -0.10, "gold": -0.03, "bond": -0.07},
                            "+100bp parallel rate shock (duration-aware)"),
        HypotheticalScenario("stagflation",
                            {"commodity": 0.15, "gold": 0.10, "bond": -0.10, "equity": -0.10},
                            "Inflationary stagnation"),
        HypotheticalScenario("risk-off-flight",
                            {"equity": -0.15, "gold": 0.08, "bond": 0.05,
                             "commodity": -0.10, "real_estate": -0.12}, "Risk-off flight to safety"),
    )


def live_stress(
    positions: dict[str, int],
    equity: float,
    *,
    asof: date,
    lookback_days: int = 180,
    scenarios=None,
) -> StressReport | None:
    """Best-effort: fetch enough history to cover the historical windows + current
    prices, then run ``compute_stress``. Returns None on flat book / data failure —
    analysis convenience, must never raise into a caller's hot path.
    """
    if not positions or equity <= 0:
        return None
    scens = tuple(scenarios) if scenarios is not None else default_scenarios()
    try:
        from quant.data.bars import BarRequest, get_bars
        from quant.strategies._common import field_frame

        symbols = sorted(set(positions))
        # Fetch from the earliest historical-scenario start (pad) through asof.
        hist_starts = [s.start for s in scens if isinstance(s, HistoricalScenario)]
        earliest = min(hist_starts) if hist_starts else asof - timedelta(days=lookback_days * 2)
        start = min(earliest, asof - timedelta(days=lookback_days * 2)) - timedelta(days=7)
        bars = get_bars(BarRequest(symbols=symbols, start=start, end=asof))
        if bars.empty:
            return None
        close = field_frame(bars, "close")
        returns = close.pct_change(fill_method=None)
        prices: dict[str, float] = {}
        for sym in close.columns:
            col = close[sym].dropna()
            if sym in positions and not col.empty:
                prices[sym] = float(col.iloc[-1])
        weights = weights_from_positions(positions, prices, equity)
        return compute_stress(weights, returns, scens)
    except Exception as exc:  # analysis convenience — never raise
        logger.info("live_stress skipped ({!r})", exc)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/risk/test_scenarios.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add quant/risk/scenarios.py tests/risk/test_scenarios.py
git commit -m "feat(risk): pure scenario/stress engine (historical + hypothetical shocks)"
```

---

### Task 2: Gate integration — `max_scenario_loss` + stress violation (pure)

**Files:**
- Modify: `quant/risk/portfolio.py` (`PortfolioRiskLimits`, `RiskGateResult`, `build_portfolio_risk_gate`)
- Test: `tests/risk/test_scenarios.py` (append gate-mapping tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/risk/test_scenarios.py`:

```python
from quant.risk.portfolio import (
    PortfolioRisk,
    PortfolioRiskLimits,
    RiskGateMode,
    build_portfolio_risk_gate,
)


def _flat_risk():
    return PortfolioRisk(
        n_positions=1, gross_exposure=1.0, net_exposure=1.0, ann_vol=0.10,
        var_95=0.01, cvar_95=0.02, beta_to_benchmark=0.5, top_name_weight=1.0,
        lookback_days=180, sector_exposure={"equity": 1.0}, computable=True,
    )


def test_stress_within_limit_no_violation():
    rep = StressReport(results=(), worst_loss=0.20, worst_scenario="x",
                       computable=True)
    gate = build_portfolio_risk_gate(_flat_risk(), limits=PortfolioRiskLimits(),
                                     mode=RiskGateMode.WARN, stress=rep)
    assert gate.ok
    assert all(v.code != "stress" for v in gate.violations)
    assert gate.stress is rep


def test_stress_over_limit_warn_violation():
    rep = StressReport(results=(), worst_loss=0.45, worst_scenario="2008-GFC",
                       computable=True)
    gate = build_portfolio_risk_gate(_flat_risk(),
                                     limits=PortfolioRiskLimits(max_scenario_loss=0.30),
                                     mode=RiskGateMode.WARN, stress=rep)
    assert not gate.ok
    assert any(v.code == "stress" for v in gate.violations)
    assert gate.severity == "warn"  # WARN never blocks


def test_stress_none_worst_never_violates():
    rep = StressReport(results=(), worst_loss=None, worst_scenario=None,
                       computable=False)
    gate = build_portfolio_risk_gate(_flat_risk(), limits=PortfolioRiskLimits(),
                                     mode=RiskGateMode.WARN, stress=rep)
    assert all(v.code != "stress" for v in gate.violations)


def test_gate_back_compat_no_stress():
    # Omitting stress preserves the prior behavior and gate.stress is None.
    gate = build_portfolio_risk_gate(_flat_risk(), limits=PortfolioRiskLimits(),
                                     mode=RiskGateMode.WARN)
    assert gate.stress is None
    assert gate.ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/risk/test_scenarios.py -q -k stress_within or stress_over or stress_none or back_compat`
Expected: FAIL — `build_portfolio_risk_gate() got an unexpected keyword argument 'stress'` and `RiskGateResult` has no `stress`.

- [ ] **Step 3: Implement**

In `quant/risk/portfolio.py`, add the limit field to `PortfolioRiskLimits` (after `max_other_bucket_weight`):

```python
    max_other_bucket_weight: float = 1.00
    max_scenario_loss: float = 0.30  # worst stress-scenario loss (positive frac); WARN cap
    fail_closed_on_uncomputable: bool = False
```

Add a `stress` field to `RiskGateResult` (after `risk: PortfolioRisk`):

```python
@dataclass(frozen=True)
class RiskGateResult:
    mode: RiskGateMode
    ok: bool
    severity: str
    violations: tuple[RiskViolation, ...]
    risk: PortfolioRisk
    stress: object | None = None  # quant.risk.scenarios.StressReport | None (duck-typed)
```

Change `build_portfolio_risk_gate`'s signature and add the stress check just before `ok = not violations`:

```python
def build_portfolio_risk_gate(
    risk: PortfolioRisk,
    *,
    limits: PortfolioRiskLimits,
    mode: RiskGateMode = RiskGateMode.WARN,
    stress: object | None = None,
) -> RiskGateResult:
```

```python
    # Stress dimension (roadmap Phase 2): worst scenario loss vs cap. Duck-typed on
    # ``stress.worst_loss`` (a quant.risk.scenarios.StressReport) to avoid an import
    # cycle. A None/absent worst_loss never violates.
    worst_loss = getattr(stress, "worst_loss", None)
    if worst_loss is not None and worst_loss > limits.max_scenario_loss:
        worst_name = getattr(stress, "worst_scenario", "?")
        violations.append(
            RiskViolation(
                "stress",
                f"stress worst-loss {worst_loss:.1%} ({worst_name}) > "
                f"{limits.max_scenario_loss:.1%}",
            )
        )

    ok = not violations
    severity = "ok" if ok else ("block" if mode is RiskGateMode.BLOCK else "warn")
    return RiskGateResult(
        mode=mode, ok=ok, severity=severity, violations=tuple(violations), risk=risk, stress=stress
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/risk/test_scenarios.py -q`
Expected: PASS (all engine + gate tests).

- [ ] **Step 5: Commit**

```bash
git add quant/risk/portfolio.py tests/risk/test_scenarios.py
git commit -m "feat(risk): thread stress worst-loss through the portfolio risk gate (WARN)"
```

---

### Task 3: Guard-5 wiring + artifact `stress` key

**Files:**
- Modify: `quant/live/rebalance.py` (`_write_portfolio_risk_gate_artifact` ~167-200, Guard 5 ~601-665)
- Test: `tests/live/test_rebalance.py`

- [ ] **Step 1: Write the failing test**

Find the existing Guard-5 / portfolio-risk-gate test in `tests/live/test_rebalance.py` (grep for `portfolio_risk_gate`). Add this test alongside it (it mirrors that test's fixtures; reuse the same fake settings / monkeypatched Alpaca + bars the existing gate test uses — match its setup exactly):

```python
def test_guard5_writes_stress_section_and_keeps_netted(monkeypatch, fake_settings, ...):
    # ... same arrange as the existing portfolio_risk_gate test (defensive-etf-like
    # book, monkeypatched broker positions + bar history) ...
    report = run_rebalance(strategy="defensive-etf-allocation", settings=fake_settings,
                           dry_run=True)
    artifact = json.loads(
        (fake_settings.data_dir / "risk"
         / f"portfolio_risk_gate.{date.today().isoformat()}.json").read_text()
    )
    # stress section present and well-formed
    assert "stress" in artifact
    assert "worst_loss" in artifact["stress"]
    assert "results" in artifact["stress"]
    # a portfolio_risk_gate CheckResult was recorded
    assert any(c.name == "portfolio_risk_gate" for c in report.safety_results)
```

> If the existing gate test asserts the exact artifact dict, update that assertion to allow the new `stress` key (additive, back-compatible).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/live/test_rebalance.py -q -k stress_section`
Expected: FAIL — `KeyError: 'stress'` (artifact has no stress key yet).

- [ ] **Step 3: Implement**

In `quant/live/rebalance.py`, extend `_write_portfolio_risk_gate_artifact` to accept and serialize stress (signature + body):

```python
def _write_portfolio_risk_gate_artifact(
    data_dir: Path, *, asof: date, gate: Any, stress: Any | None = None
) -> None:
```

Before `path = data_dir / "risk" / ...`, add the stress section when present:

```python
    if stress is not None:
        payload["stress"] = {
            "computable": stress.computable,
            "worst_loss": stress.worst_loss,
            "worst_scenario": stress.worst_scenario,
            "degraded": list(stress.degraded),
            "results": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "pnl_pct": r.pnl_pct,
                    "by_class": dict(r.by_class),
                    "missing_symbols": list(r.missing_symbols),
                    "computable": r.computable,
                }
                for r in stress.results
            ],
        }
```

In the Guard-5 block, import `live_stress`, compute it, pass into the gate + artifact. Update the import line:

```python
        from quant.risk.portfolio import (
            PortfolioRisk,
            PortfolioRiskLimits,
            RiskGateMode,
            build_portfolio_risk_gate,
            live_portfolio_risk,
        )
        from quant.risk.scenarios import live_stress
```

After `port_risk` is finalized (the `if port_risk is None:` fallback block) and before `gate = build_portfolio_risk_gate(...)`, compute stress and thread it through:

```python
        stress = live_stress(post_trade, account.equity, asof=asof)
        gate = build_portfolio_risk_gate(
            port_risk, limits=PortfolioRiskLimits(), mode=gate_mode, stress=stress
        )
        safety_results.append(
            CheckResult(ok=gate.ok, name="portfolio_risk_gate", detail=gate.detail)
        )
        _write_portfolio_risk_gate_artifact(settings.data_dir, asof=asof, gate=gate, stress=stress)
```

(`netted` is not referenced here — it stays byte-identical. The existing fail-open `try/except` already wraps this block.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/live/test_rebalance.py -q`
Expected: PASS (20+ tests, including the new one and the existing dry-run/gate tests).

- [ ] **Step 5: Commit**

```bash
git add quant/live/rebalance.py tests/live/test_rebalance.py
git commit -m "feat(live): Guard 5 computes stress + folds it into the risk-gate artifact (WARN)"
```

---

### Task 4: CLI `quant risk scenarios`

**Files:**
- Modify: `quant/cli.py` (add command after `risk_portfolio`, ~line 2110)
- Test: `tests/test_cli.py` (or the module containing the `risk portfolio` CLI test — grep for `risk_portfolio` / `"portfolio"`)

- [ ] **Step 1: Write the failing test**

Add to the CLI test module (mirror the existing `risk portfolio` CLI test's invocation + Alpaca monkeypatch; use `CliRunner`):

```python
def test_risk_scenarios_flat_book(monkeypatch):
    from click.testing import CliRunner
    from quant.cli import cli
    # monkeypatch AlpacaClient so positions() -> [] (flat book), account().equity -> 0
    # ... (match the existing risk-portfolio CLI test's monkeypatch style) ...
    result = CliRunner().invoke(cli, ["risk", "scenarios"])
    assert result.exit_code == 0
    assert "flat" in result.output.lower() or "no " in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest <cli test path> -q -k risk_scenarios`
Expected: FAIL — no such command `scenarios` (Click usage error / non-zero exit).

- [ ] **Step 3: Implement**

In `quant/cli.py`, immediately after the `risk_portfolio` function, add:

```python
@risk.command(
    "scenarios",
    help="Stress the LIVE book under historical + hypothetical shock scenarios. Read-only.",
)
@click.option("--lookback", default=180, show_default=True, type=int,
              help="Trading-day window for current weights.")
def risk_scenarios(lookback: int) -> None:
    from quant.risk.scenarios import live_stress

    settings = Settings()  # type: ignore[call-arg]
    asof = date.today()
    positions: dict[str, int] = {}
    equity = 0.0
    try:
        client = AlpacaClient(settings=settings)
        equity = float(client.account().equity)
        positions = {p.symbol: int(p.qty) for p in client.positions()}
    except Exception as exc:
        raise click.ClickException(f"Alpaca unavailable: {exc!r}") from exc

    if not positions:
        console.print("[yellow]Book is flat — no scenarios to stress.[/yellow]")
        return

    rep = live_stress(positions, equity, asof=asof, lookback_days=lookback)
    if rep is None or not rep.computable:
        console.print("[yellow]Could not compute stress (no bar history?).[/yellow]")
        return

    out = settings.data_dir / "risk" / f"scenarios.{asof.isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "asof": asof.isoformat(),
                "equity": equity,
                "worst_loss": rep.worst_loss,
                "worst_scenario": rep.worst_scenario,
                "degraded": list(rep.degraded),
                "results": [
                    {"name": r.name, "kind": r.kind, "pnl_pct": r.pnl_pct,
                     "by_class": dict(r.by_class), "missing_symbols": list(r.missing_symbols),
                     "computable": r.computable}
                    for r in rep.results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table = Table(title=f"stress scenarios — {asof.isoformat()}")
    table.add_column("Scenario")
    table.add_column("Kind")
    table.add_column("P&L", justify="right")
    for r in rep.results:
        pnl = "n/a" if r.pnl_pct is None else f"{r.pnl_pct:+.1%}"
        style = "red" if (r.pnl_pct is not None and r.pnl_pct < 0) else "green"
        marker = " ◀ worst" if r.name == rep.worst_scenario else ""
        table.add_row(r.name, r.kind, f"[{style}]{pnl}[/{style}]{marker}")
    console.print(table)
    console.print(rep.render())
    console.print(f"[dim]wrote {out}[/dim]")
```

> Confirm `Table` is already imported in `cli.py` (the `risk_portfolio`/other commands use `rich`). If `Table` is not imported, add `from rich.table import Table` at the top with the other imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest <cli test path> -q -k risk_scenarios`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/cli.py <cli test path>
git commit -m "feat(cli): quant risk scenarios — stress the live book (read-only)"
```

---

### Task 5: Analyst brief surfacing

**Files:**
- Modify: `quant/analyst/context.py` (`AnalystContext`, `_stress` helper, `gather_analyst_context`, render block ~323)
- Test: `tests/analyst/test_context.py` (grep for the existing portfolio-risk brief test)

- [ ] **Step 1: Write the failing test**

Add to the analyst context test module (mirror the existing `portfolio_risk` brief test):

```python
def test_brief_includes_stress_line():
    from quant.analyst.context import AnalystContext, render_brief  # render fn name per module
    from quant.risk.scenarios import StressReport, ScenarioResult
    rep = StressReport(
        results=(ScenarioResult("2008-GFC", "historical", -0.18),),
        worst_loss=0.18, worst_scenario="2008-GFC", computable=True,
    )
    ctx = AnalystContext(asof=date(2026, 6, 5), stress=rep)
    text = render_brief(ctx)  # use the actual render function used for portfolio_risk
    assert "Stress" in text
    assert "2008-GFC" in text
```

> Match the real render function name used in `context.py` for the brief (the function containing the `"Portfolio risk: " + ctx.portfolio_risk.render()` line). Adjust the import/call accordingly.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/analyst/test_context.py -q -k stress_line`
Expected: FAIL — `AnalystContext.__init__() got an unexpected keyword argument 'stress'`.

- [ ] **Step 3: Implement**

In `quant/analyst/context.py`, add the field to `AnalystContext` (after `portfolio_risk`):

```python
    portfolio_risk: Any | None = None  # quant.risk.PortfolioRisk | None (lazy to avoid cycle)
    stress: Any | None = None  # quant.risk.scenarios.StressReport | None (lazy)
```

Add a helper next to `_portfolio_risk`:

```python
def _stress(
    positions: dict[str, int] | None, equity: float | None, asof: date
) -> Any | None:
    if not positions or not equity or equity <= 0:
        return None
    try:
        from quant.risk.scenarios import live_stress

        return live_stress(positions, float(equity), asof=asof)
    except Exception as exc:  # fail-open
        logger.info("analyst.context: stress skipped ({!r})", exc)
        return None
```

Wire it in `gather_analyst_context`'s returned `AnalystContext(...)` (next to `portfolio_risk=...`):

```python
        portfolio_risk=_portfolio_risk(positions, equity, asof),
        stress=_stress(positions, equity, asof),
```

Add the render line right after the `portfolio_risk` render block (~line 325):

```python
    if ctx.stress is not None:
        with contextlib.suppress(Exception):  # render is best-effort
            lines.append("Stress: " + ctx.stress.render())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/analyst/test_context.py -q -k stress_line`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quant/analyst/context.py tests/analyst/test_context.py
git commit -m "feat(analyst): surface stress worst-loss in the read-only brief"
```

---

### Task 6: Full-suite green + calibrate `max_scenario_loss` against the live book

**Files:**
- Possibly modify: `quant/risk/portfolio.py` (`max_scenario_loss` default, only if the live worst-case is close)

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (no regressions; new stress/CLI/brief/rebalance tests green).

- [ ] **Step 2: Lint/format**

Run: `uv run ruff check quant/risk/scenarios.py quant/cli.py quant/analyst/context.py quant/live/rebalance.py && uv run ruff format --check quant/risk/scenarios.py`
Expected: clean (run `uv run ruff format <files>` if needed, then re-commit).

- [ ] **Step 3: Calibrate against the real defensive book**

Run: `uv run quant risk scenarios` (needs Alpaca paper creds in `.env`; if unavailable in this worktree, run from the main checkout's env or skip with a note).
Inspect the worst-loss. If the live defensive book's worst scenario loss is within ~20% of the `0.30` default (i.e. > ~0.24), widen `max_scenario_loss` so defensive-etf records OK with headroom, and note the measured worst-case in a code comment. If comfortably below, leave `0.30`.

Expected: defensive-etf book records `portfolio_risk_gate` OK (no `stress` violation) at the chosen default.

- [ ] **Step 4: Commit any calibration change**

```bash
git add quant/risk/portfolio.py
git commit -m "chore(risk): calibrate max_scenario_loss to the live defensive sleeve"
```

(Skip this commit if no change was needed.)

---

## Self-Review Notes

- **Spec coverage:** module (Task 1) ✓, gate/limit (Task 2) ✓, Guard-5 + artifact (Task 3) ✓, CLI (Task 4) ✓, brief (Task 5) ✓, calibration + suite (Task 6) ✓. Out-of-scope items (BLOCK flip, factor model) intentionally absent.
- **Type consistency:** `StressReport`/`ScenarioResult` field names (`pnl_pct`, `worst_loss`, `worst_scenario`, `by_class`, `missing_symbols`, `computable`, `degraded`) are used identically across the module, gate, artifact, CLI, and brief. `build_portfolio_risk_gate(..., stress=...)` and `RiskGateResult.stress` match between Task 2 and Task 3. `live_stress(positions, equity, *, asof, lookback_days=, scenarios=)` signature consistent across module/Guard-5/CLI/brief.
- **Duck-typing:** the gate accesses `stress.worst_loss`/`.worst_scenario` only via `getattr`, so `portfolio.py` never imports `scenarios.py` (no cycle); `scenarios.py` imports only `_SECTOR_MAP`/`weights_from_positions` from `portfolio.py`.
- **Placeholder note:** Tasks 3-5 reference "the existing test's fixtures/monkeypatch" rather than reproducing unseen fixture code; the implementer must open the named test module and match its established setup. This is deliberate (those fixtures are codebase-specific and must be reused, not reinvented), not a content gap in the new code.
