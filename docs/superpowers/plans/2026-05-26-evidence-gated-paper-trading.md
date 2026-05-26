# Evidence-Gated Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed strategy governance layer so normal paper rebalances allocate only to strategies with fresh passing evidence.

**Architecture:** Governance is a pure package under `quant/governance/` with typed dataclasses, deterministic JSON stores, and a policy classifier. `quant validate` writes a machine-readable validation sidecar; `quant governance refresh` converts validation artifacts plus registry metadata into `data/governance/*.json`; `quant rebalance` filters through those states by default.

**Tech Stack:** Python 3.12, dataclasses, Click, Rich, pandas/parquet, pytest, existing `Settings.data_dir`, existing strategy registry.

---

## File Structure

- Create `quant/governance/__init__.py`
  Public exports for governance models, loaders, refresh helpers, and errors.

- Create `quant/governance/models.py`
  Dataclasses and enums for `GovernanceState`, `ValidationEvidence`, `StrategyState`, `GovernancePolicy`, and `GovernanceError`.

- Create `quant/governance/store.py`
  Deterministic JSON read/write helpers for `validation_manifest.json`, `strategy_states.json`, and malformed/missing artifact handling.

- Create `quant/governance/policy.py`
  Pure classifier that maps registry specs + evidence + artifact existence to `live`, `quarantined`, or `research`.

- Create `quant/governance/refresh.py`
  Reads `data/backtests/<slug>/validation_report.json`, `chosen_params.json`, and `walkforward.parquet`; writes governance artifacts.

- Modify `quant/cli.py`
  Add `quant governance status`, `quant governance refresh`, write validation sidecar from `quant validate`, show governance state in `quant strategies`, and pass governance options into rebalance.

- Modify `quant/live/rebalance.py`
  Add governance filtering for default strategy selection and dry-run-only quarantined inclusion.

- Create `tests/governance/test_policy.py`
  Unit tests for state classification.

- Create `tests/governance/test_store.py`
  Unit tests for deterministic JSON and fail-closed malformed data.

- Create `tests/governance/test_refresh.py`
  Unit tests for manifest/state generation from fake validation artifacts.

- Modify `tests/test_cli.py`
  CLI coverage for governance command group and strategy table state.

- Modify `tests/live/test_rebalance.py`
  Rebalance coverage for governance filtering and dry-run quarantined observation.

- Modify `tests/conftest.py`
  Add `data/governance` to temporary data directories.

- Modify `README.md`
  Document the new governance workflow and fail-closed rebalance behavior.

---

### Task 1: Governance Models And Policy

**Files:**
- Create: `quant/governance/__init__.py`
- Create: `quant/governance/models.py`
- Create: `quant/governance/policy.py`
- Create: `tests/governance/test_policy.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add governance data dir to test fixture**

In `tests/conftest.py`, change the `tmp_data_dir` fixture subdirectory tuple to include `governance`:

```python
for sub in (
    "universe",
    "raw",
    "backtests",
    "live",
    "features",
    "fundamentals",
    "macro",
    "governance",
):
    (data / sub).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Write failing policy tests**

Create `tests/governance/test_policy.py`:

```python
"""Tests for evidence-gated strategy governance policy."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant.governance.models import GovernancePolicy, GovernanceState, ValidationEvidence
from quant.governance.policy import classify_strategy
from quant.strategies.base import StrategySpec


def _spec(*, enabled_live: bool = True) -> StrategySpec:
    return StrategySpec(
        slug="trend",
        name="Trend",
        description="",
        universe=["SPY"],
        rebalance_frequency="monthly",
        enabled_live=enabled_live,
    )


def _evidence(**overrides: object) -> ValidationEvidence:
    values: dict[str, object] = {
        "slug": "trend",
        "run_date": date(2026, 5, 20),
        "data_start": date(2010, 1, 1),
        "data_end": date(2026, 5, 19),
        "gate_deflated_sharpe": True,
        "gate_probabilistic_sharpe": True,
        "gate_bootstrap_lower": True,
        "gate_regime": True,
        "gate_holdout": True,
        "deflated_sharpe": 0.54,
        "probabilistic_sharpe": 0.99,
        "bootstrap_total_return_p05": 0.12,
        "n_positive_regimes": 3,
        "n_tested_regimes": 3,
        "holdout_total_return": 0.21,
        "chosen_params_path": "data/backtests/trend/chosen_params.json",
        "walkforward_path": "data/backtests/trend/walkforward.parquet",
        "provenance": "unit test",
    }
    values.update(overrides)
    return ValidationEvidence(**values)


def test_live_when_enabled_fresh_all_gates_pass_and_artifacts_exist(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.LIVE
    assert state.reason_codes == []


def test_missing_evidence_quarantines_live_capable_strategy() -> None:
    state = classify_strategy(
        spec=_spec(),
        evidence=None,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "missing_validation" in state.reason_codes


def test_disabled_strategy_is_research_even_without_evidence() -> None:
    state = classify_strategy(
        spec=_spec(enabled_live=False),
        evidence=None,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.RESEARCH
    assert "not_live_capable" in state.reason_codes


def test_stale_evidence_quarantines_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        run_date=date(2026, 4, 1),
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "stale_validation" in state.reason_codes


def test_failed_gate_quarantines_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        gate_regime=False,
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "failed_gate_regime" in state.reason_codes


def test_manual_block_quarantines_otherwise_passing_strategy(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen_params.json"
    walkforward = tmp_path / "walkforward.parquet"
    chosen.write_text("{}")
    walkforward.write_text("fake parquet")
    evidence = _evidence(
        chosen_params_path=str(chosen),
        walkforward_path=str(walkforward),
        manual_block=True,
        manual_block_reason="paper drawdown review",
    )
    state = classify_strategy(
        spec=_spec(),
        evidence=evidence,
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert state.state is GovernanceState.QUARANTINED
    assert "manual_block" in state.reason_codes
    assert "paper drawdown review" in state.reason
```

- [ ] **Step 3: Run policy tests and verify they fail**

Run:

```bash
uv run pytest tests/governance/test_policy.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'quant.governance'`.

- [ ] **Step 4: Add governance models**

Create `quant/governance/models.py`:

```python
"""Typed models for strategy governance artifacts and decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class GovernanceError(RuntimeError):
    """Raised when governance artifacts are missing, stale, or malformed."""


class GovernanceState(StrEnum):
    LIVE = "live"
    QUARANTINED = "quarantined"
    RESEARCH = "research"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GovernancePolicy:
    max_validation_age_days: int = 30
    require_deflated_sharpe: bool = True
    require_probabilistic_sharpe: bool = True
    require_bootstrap_lower: bool = True
    require_regime: bool = True
    require_holdout: bool = True


@dataclass(frozen=True)
class ValidationEvidence:
    slug: str
    run_date: date
    data_start: date
    data_end: date
    gate_deflated_sharpe: bool
    gate_probabilistic_sharpe: bool
    gate_bootstrap_lower: bool
    gate_regime: bool
    gate_holdout: bool
    deflated_sharpe: float
    probabilistic_sharpe: float
    bootstrap_total_return_p05: float | None
    n_positive_regimes: int
    n_tested_regimes: int
    holdout_total_return: float | None
    chosen_params_path: str
    walkforward_path: str
    provenance: str
    manual_block: bool = False
    manual_block_reason: str | None = None

    def gate_map(self) -> dict[str, bool]:
        return {
            "deflated_sharpe": self.gate_deflated_sharpe,
            "probabilistic_sharpe": self.gate_probabilistic_sharpe,
            "bootstrap_lower": self.gate_bootstrap_lower,
            "regime": self.gate_regime,
            "holdout": self.gate_holdout,
        }

    def artifact_paths(self) -> tuple[Path, Path]:
        return Path(self.chosen_params_path), Path(self.walkforward_path)


@dataclass(frozen=True)
class StrategyState:
    slug: str
    state: GovernanceState
    evaluated_at: datetime
    validation_age_days: int | None
    reason_codes: list[str] = field(default_factory=list)
    reason: str = ""
    code_enabled_live: bool = False
    manual_block: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["evaluated_at"] = self.evaluated_at.isoformat()
        return payload
```

Create `quant/governance/__init__.py`:

```python
"""Strategy governance: evidence-gated paper-trading eligibility."""

from quant.governance.models import (
    GovernanceError,
    GovernancePolicy,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)
from quant.governance.policy import classify_strategy

__all__ = [
    "GovernanceError",
    "GovernancePolicy",
    "GovernanceState",
    "StrategyState",
    "ValidationEvidence",
    "classify_strategy",
]
```

- [ ] **Step 5: Add policy classifier**

Create `quant/governance/policy.py`:

```python
"""Pure strategy-governance classification rules."""

from __future__ import annotations

from datetime import date, datetime

from quant.governance.models import (
    GovernancePolicy,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)
from quant.strategies.base import StrategySpec


def classify_strategy(
    *,
    spec: StrategySpec,
    evidence: ValidationEvidence | None,
    policy: GovernancePolicy,
    asof: date,
) -> StrategyState:
    reason_codes: list[str] = []
    reason_parts: list[str] = []
    validation_age_days: int | None = None

    if not spec.enabled_live:
        return StrategyState(
            slug=spec.slug,
            state=GovernanceState.RESEARCH,
            evaluated_at=datetime.combine(asof, datetime.min.time()),
            validation_age_days=None,
            reason_codes=["not_live_capable"],
            reason="StrategySpec.enabled_live is false; research only.",
            code_enabled_live=False,
        )

    if evidence is None:
        return StrategyState(
            slug=spec.slug,
            state=GovernanceState.QUARANTINED,
            evaluated_at=datetime.combine(asof, datetime.min.time()),
            validation_age_days=None,
            reason_codes=["missing_validation"],
            reason="No validation evidence exists for this live-capable strategy.",
            code_enabled_live=True,
        )

    validation_age_days = (asof - evidence.run_date).days
    if validation_age_days < 0:
        reason_codes.append("future_validation_date")
        reason_parts.append(f"Validation run date {evidence.run_date} is after {asof}.")
    if validation_age_days > policy.max_validation_age_days:
        reason_codes.append("stale_validation")
        reason_parts.append(
            f"Validation is {validation_age_days} days old; limit is "
            f"{policy.max_validation_age_days} days."
        )

    gate_requirements = {
        "deflated_sharpe": policy.require_deflated_sharpe,
        "probabilistic_sharpe": policy.require_probabilistic_sharpe,
        "bootstrap_lower": policy.require_bootstrap_lower,
        "regime": policy.require_regime,
        "holdout": policy.require_holdout,
    }
    for gate, required in gate_requirements.items():
        if required and not evidence.gate_map()[gate]:
            reason_codes.append(f"failed_gate_{gate}")
            reason_parts.append(f"Required gate failed: {gate}.")

    chosen_path, walkforward_path = evidence.artifact_paths()
    if not chosen_path.exists():
        reason_codes.append("missing_chosen_params")
        reason_parts.append(f"Missing chosen params artifact: {chosen_path}.")
    if not walkforward_path.exists():
        reason_codes.append("missing_walkforward")
        reason_parts.append(f"Missing walk-forward artifact: {walkforward_path}.")

    if evidence.manual_block:
        reason_codes.append("manual_block")
        block_reason = evidence.manual_block_reason or "manual block is active"
        reason_parts.append(f"Manual block: {block_reason}.")

    state = GovernanceState.LIVE if not reason_codes else GovernanceState.QUARANTINED
    reason = "Fresh validation evidence passes all required gates." if state is GovernanceState.LIVE else " ".join(reason_parts)
    return StrategyState(
        slug=spec.slug,
        state=state,
        evaluated_at=datetime.combine(asof, datetime.min.time()),
        validation_age_days=validation_age_days,
        reason_codes=reason_codes,
        reason=reason,
        code_enabled_live=spec.enabled_live,
        manual_block=evidence.manual_block,
    )
```

- [ ] **Step 6: Run policy tests and verify they pass**

Run:

```bash
uv run pytest tests/governance/test_policy.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add quant/governance/__init__.py quant/governance/models.py quant/governance/policy.py tests/governance/test_policy.py tests/conftest.py
git commit -m "feat(governance): classify strategy eligibility"
```

---

### Task 2: Deterministic Governance Stores

**Files:**
- Create: `quant/governance/store.py`
- Create: `tests/governance/test_store.py`
- Modify: `quant/governance/__init__.py`

- [ ] **Step 1: Write failing store tests**

Create `tests/governance/test_store.py`:

```python
"""Tests for governance JSON artifact stores."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from quant.governance.models import GovernanceError, GovernanceState, StrategyState, ValidationEvidence
from quant.governance.store import (
    load_strategy_states,
    load_validation_manifest,
    write_strategy_states,
    write_validation_manifest,
)


def _evidence(slug: str = "trend") -> ValidationEvidence:
    return ValidationEvidence(
        slug=slug,
        run_date=date(2026, 5, 20),
        data_start=date(2010, 1, 1),
        data_end=date(2026, 5, 19),
        gate_deflated_sharpe=True,
        gate_probabilistic_sharpe=True,
        gate_bootstrap_lower=True,
        gate_regime=True,
        gate_holdout=True,
        deflated_sharpe=0.54,
        probabilistic_sharpe=0.99,
        bootstrap_total_return_p05=0.12,
        n_positive_regimes=3,
        n_tested_regimes=3,
        holdout_total_return=0.21,
        chosen_params_path="data/backtests/trend/chosen_params.json",
        walkforward_path="data/backtests/trend/walkforward.parquet",
        provenance="unit test",
    )


def test_validation_manifest_round_trips_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    write_validation_manifest(path, {"trend": _evidence()})
    first = path.read_text()
    write_validation_manifest(path, {"trend": _evidence()})
    second = path.read_text()
    loaded = load_validation_manifest(path)
    assert first == second
    assert loaded["trend"].run_date == date(2026, 5, 20)
    assert json.loads(first)["strategies"]["trend"]["slug"] == "trend"


def test_strategy_states_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "strategy_states.json"
    state = StrategyState(
        slug="trend",
        state=GovernanceState.LIVE,
        evaluated_at=datetime(2026, 5, 26),
        validation_age_days=6,
        reason_codes=[],
        reason="ok",
        code_enabled_live=True,
    )
    write_strategy_states(path, {"trend": state})
    loaded = load_strategy_states(path)
    assert loaded["trend"].state is GovernanceState.LIVE
    assert loaded["trend"].validation_age_days == 6


def test_missing_manifest_raises_governance_error(tmp_path: Path) -> None:
    with pytest.raises(GovernanceError, match="Missing governance artifact"):
        load_validation_manifest(tmp_path / "missing.json")


def test_malformed_manifest_raises_governance_error(tmp_path: Path) -> None:
    path = tmp_path / "validation_manifest.json"
    path.write_text("{not-json")
    with pytest.raises(GovernanceError, match="Malformed governance artifact"):
        load_validation_manifest(path)
```

- [ ] **Step 2: Run store tests and verify they fail**

Run:

```bash
uv run pytest tests/governance/test_store.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `quant.governance.store`.

- [ ] **Step 3: Implement store helpers**

Create `quant/governance/store.py`:

```python
"""Deterministic JSON stores for strategy governance artifacts."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from quant.governance.models import (
    GovernanceError,
    GovernanceState,
    StrategyState,
    ValidationEvidence,
)


VALIDATION_MANIFEST_NAME = "validation_manifest.json"
STRATEGY_STATES_NAME = "strategy_states.json"


def governance_dir(data_dir: Path) -> Path:
    return data_dir / "governance"


def validation_manifest_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / VALIDATION_MANIFEST_NAME


def strategy_states_path(data_dir: Path) -> Path:
    return governance_dir(data_dir) / STRATEGY_STATES_NAME


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GovernanceError(f"Missing governance artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GovernanceError(f"Malformed governance artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    return payload


def write_validation_manifest(path: Path, evidence_by_slug: dict[str, ValidationEvidence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: {
                "slug": evidence.slug,
                "run_date": evidence.run_date.isoformat(),
                "data_start": evidence.data_start.isoformat(),
                "data_end": evidence.data_end.isoformat(),
                "gate_deflated_sharpe": evidence.gate_deflated_sharpe,
                "gate_probabilistic_sharpe": evidence.gate_probabilistic_sharpe,
                "gate_bootstrap_lower": evidence.gate_bootstrap_lower,
                "gate_regime": evidence.gate_regime,
                "gate_holdout": evidence.gate_holdout,
                "deflated_sharpe": evidence.deflated_sharpe,
                "probabilistic_sharpe": evidence.probabilistic_sharpe,
                "bootstrap_total_return_p05": evidence.bootstrap_total_return_p05,
                "n_positive_regimes": evidence.n_positive_regimes,
                "n_tested_regimes": evidence.n_tested_regimes,
                "holdout_total_return": evidence.holdout_total_return,
                "chosen_params_path": evidence.chosen_params_path,
                "walkforward_path": evidence.walkforward_path,
                "provenance": evidence.provenance,
                "manual_block": evidence.manual_block,
                "manual_block_reason": evidence.manual_block_reason,
            }
            for slug, evidence in sorted(evidence_by_slug.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_validation_manifest(path: Path) -> dict[str, ValidationEvidence]:
    payload = _load_json(path)
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, ValidationEvidence] = {}
    for slug, raw in strategies.items():
        if not isinstance(raw, dict):
            raise GovernanceError(f"Malformed governance artifact: {path}")
        out[str(slug)] = ValidationEvidence(
            slug=str(raw["slug"]),
            run_date=_date(str(raw["run_date"])),
            data_start=_date(str(raw["data_start"])),
            data_end=_date(str(raw["data_end"])),
            gate_deflated_sharpe=bool(raw["gate_deflated_sharpe"]),
            gate_probabilistic_sharpe=bool(raw["gate_probabilistic_sharpe"]),
            gate_bootstrap_lower=bool(raw["gate_bootstrap_lower"]),
            gate_regime=bool(raw["gate_regime"]),
            gate_holdout=bool(raw["gate_holdout"]),
            deflated_sharpe=float(raw["deflated_sharpe"]),
            probabilistic_sharpe=float(raw["probabilistic_sharpe"]),
            bootstrap_total_return_p05=(
                None
                if raw.get("bootstrap_total_return_p05") is None
                else float(raw["bootstrap_total_return_p05"])
            ),
            n_positive_regimes=int(raw["n_positive_regimes"]),
            n_tested_regimes=int(raw["n_tested_regimes"]),
            holdout_total_return=(
                None if raw.get("holdout_total_return") is None else float(raw["holdout_total_return"])
            ),
            chosen_params_path=str(raw["chosen_params_path"]),
            walkforward_path=str(raw["walkforward_path"]),
            provenance=str(raw["provenance"]),
            manual_block=bool(raw.get("manual_block", False)),
            manual_block_reason=(
                None if raw.get("manual_block_reason") is None else str(raw["manual_block_reason"])
            ),
        )
    return out


def write_strategy_states(path: Path, states_by_slug: dict[str, StrategyState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: state.to_json_dict() for slug, state in sorted(states_by_slug.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_strategy_states(path: Path) -> dict[str, StrategyState]:
    payload = _load_json(path)
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        raise GovernanceError(f"Malformed governance artifact: {path}")
    out: dict[str, StrategyState] = {}
    for slug, raw in strategies.items():
        if not isinstance(raw, dict):
            raise GovernanceError(f"Malformed governance artifact: {path}")
        out[str(slug)] = StrategyState(
            slug=str(raw["slug"]),
            state=GovernanceState(str(raw["state"])),
            evaluated_at=_datetime(str(raw["evaluated_at"])),
            validation_age_days=(
                None if raw.get("validation_age_days") is None else int(raw["validation_age_days"])
            ),
            reason_codes=[str(x) for x in raw.get("reason_codes", [])],
            reason=str(raw.get("reason", "")),
            code_enabled_live=bool(raw.get("code_enabled_live", False)),
            manual_block=bool(raw.get("manual_block", False)),
        )
    return out
```

- [ ] **Step 4: Export store helpers**

Modify `quant/governance/__init__.py` to include:

```python
from quant.governance.store import (
    governance_dir,
    load_strategy_states,
    load_validation_manifest,
    strategy_states_path,
    validation_manifest_path,
    write_strategy_states,
    write_validation_manifest,
)
```

Add these names to `__all__`.

- [ ] **Step 5: Run store tests and verify they pass**

Run:

```bash
uv run pytest tests/governance/test_store.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add quant/governance/__init__.py quant/governance/store.py tests/governance/test_store.py
git commit -m "feat(governance): add JSON artifact stores"
```

---

### Task 3: Validation Sidecar And Governance Refresh

**Files:**
- Create: `quant/governance/refresh.py`
- Create: `tests/governance/test_refresh.py`
- Modify: `quant/cli.py`
- Modify: `quant/governance/__init__.py`

- [ ] **Step 1: Write failing refresh tests**

Create `tests/governance/test_refresh.py`:

```python
"""Tests for generating governance artifacts from validation sidecars."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quant.governance.models import GovernancePolicy, GovernanceState
from quant.governance.refresh import build_governance_artifacts, validation_report_to_evidence
from quant.strategies import REGISTRY
from quant.strategies.base import StrategySpec


def _write_validation_artifacts(data_dir: Path, slug: str, *, gate_regime: bool = True) -> None:
    out = data_dir / "backtests" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "chosen_params.json").write_text(json.dumps({"latest": {"x": 1}, "windows": []}))
    (out / "walkforward.parquet").write_text("fake parquet")
    (out / "validation_report.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "run_date": "2026-05-20",
                "data_start": "2010-01-01",
                "data_end": "2026-05-19",
                "gate_deflated_sharpe": True,
                "gate_probabilistic_sharpe": True,
                "gate_bootstrap_lower": True,
                "gate_regime": gate_regime,
                "gate_holdout": True,
                "deflated_sharpe": 0.54,
                "probabilistic_sharpe": 0.99,
                "bootstrap_total_return_p05": 0.12,
                "n_positive_regimes": 3,
                "n_tested_regimes": 3,
                "holdout_total_return": 0.21,
                "provenance": "uv run quant validate trend",
            }
        )
    )


def test_validation_report_to_evidence_adds_artifact_paths(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")
    evidence = validation_report_to_evidence(tmp_data_dir, "trend")
    assert evidence is not None
    assert evidence.slug == "trend"
    assert evidence.chosen_params_path.endswith("chosen_params.json")
    assert evidence.walkforward_path.endswith("walkforward.parquet")


def test_build_governance_artifacts_classifies_registry(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend")
    states = build_governance_artifacts(
        data_dir=tmp_data_dir,
        registry={"trend": type("S", (), {"spec": StrategySpec(
            slug="trend",
            name="Trend",
            description="",
            universe=["SPY"],
            rebalance_frequency="monthly",
            enabled_live=True,
        )})},
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert states["trend"].state is GovernanceState.LIVE
    assert (tmp_data_dir / "governance" / "validation_manifest.json").exists()
    assert (tmp_data_dir / "governance" / "strategy_states.json").exists()


def test_failed_validation_report_quarantines(tmp_data_dir: Path) -> None:
    _write_validation_artifacts(tmp_data_dir, "trend", gate_regime=False)
    states = build_governance_artifacts(
        data_dir=tmp_data_dir,
        registry={"trend": type("S", (), {"spec": REGISTRY["trend"].spec})},
        policy=GovernancePolicy(max_validation_age_days=30),
        asof=date(2026, 5, 26),
    )
    assert states["trend"].state is GovernanceState.QUARANTINED
    assert "failed_gate_regime" in states["trend"].reason_codes
```

- [ ] **Step 2: Run refresh tests and verify they fail**

Run:

```bash
uv run pytest tests/governance/test_refresh.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `refresh` functions.

- [ ] **Step 3: Implement governance refresh helpers**

Create `quant/governance/refresh.py`:

```python
"""Build governance artifacts from validation sidecars and the strategy registry."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from quant.governance.models import GovernancePolicy, StrategyState, ValidationEvidence
from quant.governance.policy import classify_strategy
from quant.governance.store import (
    strategy_states_path,
    validation_manifest_path,
    write_strategy_states,
    write_validation_manifest,
)


def validation_report_path(data_dir: Path, slug: str) -> Path:
    return data_dir / "backtests" / slug / "validation_report.json"


def _read_validation_report(data_dir: Path, slug: str) -> dict[str, Any] | None:
    path = validation_report_path(data_dir, slug)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def validation_report_to_evidence(data_dir: Path, slug: str) -> ValidationEvidence | None:
    raw = _read_validation_report(data_dir, slug)
    if raw is None:
        return None
    backtest_dir = data_dir / "backtests" / slug
    return ValidationEvidence(
        slug=str(raw["slug"]),
        run_date=date.fromisoformat(str(raw["run_date"])),
        data_start=date.fromisoformat(str(raw["data_start"])),
        data_end=date.fromisoformat(str(raw["data_end"])),
        gate_deflated_sharpe=bool(raw["gate_deflated_sharpe"]),
        gate_probabilistic_sharpe=bool(raw["gate_probabilistic_sharpe"]),
        gate_bootstrap_lower=bool(raw["gate_bootstrap_lower"]),
        gate_regime=bool(raw["gate_regime"]),
        gate_holdout=bool(raw["gate_holdout"]),
        deflated_sharpe=float(raw["deflated_sharpe"]),
        probabilistic_sharpe=float(raw["probabilistic_sharpe"]),
        bootstrap_total_return_p05=(
            None
            if raw.get("bootstrap_total_return_p05") is None
            else float(raw["bootstrap_total_return_p05"])
        ),
        n_positive_regimes=int(raw["n_positive_regimes"]),
        n_tested_regimes=int(raw["n_tested_regimes"]),
        holdout_total_return=(
            None if raw.get("holdout_total_return") is None else float(raw["holdout_total_return"])
        ),
        chosen_params_path=str(backtest_dir / "chosen_params.json"),
        walkforward_path=str(backtest_dir / "walkforward.parquet"),
        provenance=str(raw.get("provenance", f"validation_report:{validation_report_path(data_dir, slug)}")),
        manual_block=bool(raw.get("manual_block", False)),
        manual_block_reason=(
            None if raw.get("manual_block_reason") is None else str(raw["manual_block_reason"])
        ),
    )


def build_governance_artifacts(
    *,
    data_dir: Path,
    registry: dict[str, Any],
    policy: GovernancePolicy,
    asof: date,
) -> dict[str, StrategyState]:
    evidence_by_slug: dict[str, ValidationEvidence] = {}
    states: dict[str, StrategyState] = {}
    for slug, strategy_cls in sorted(registry.items()):
        evidence = validation_report_to_evidence(data_dir, slug)
        if evidence is not None:
            evidence_by_slug[slug] = evidence
        states[slug] = classify_strategy(
            spec=strategy_cls.spec,
            evidence=evidence,
            policy=policy,
            asof=asof,
        )
    write_validation_manifest(validation_manifest_path(data_dir), evidence_by_slug)
    write_strategy_states(strategy_states_path(data_dir), states)
    return states
```

- [ ] **Step 4: Export refresh helpers**

Modify `quant/governance/__init__.py` to include:

```python
from quant.governance.refresh import (
    build_governance_artifacts,
    validation_report_path,
    validation_report_to_evidence,
)
```

Add those names to `__all__`.

- [ ] **Step 5: Make `quant validate` write `validation_report.json`**

In `quant/cli.py`, add this helper near the validation command:

```python
def _write_validation_report_json(
    *,
    out_dir: Path,
    slug: str,
    run_date: date,
    data_start: date,
    data_end: date,
    report: Any,
    provenance: str,
) -> Path:
    import json

    n_tested = sum(1 for r in report.regime_breakdown if r.n_days >= 30)
    payload = {
        "slug": slug,
        "run_date": run_date.isoformat(),
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
        "gate_deflated_sharpe": bool(report.gate_deflated_sharpe),
        "gate_probabilistic_sharpe": bool(report.gate_probabilistic_sharpe),
        "gate_bootstrap_lower": bool(report.gate_bootstrap_lower),
        "gate_regime": bool(report.gate_regime),
        "gate_holdout": bool(report.gate_holdout),
        "deflated_sharpe": float(report.deflated_sharpe),
        "probabilistic_sharpe": float(report.probabilistic_sharpe),
        "bootstrap_total_return_p05": (
            None if report.bootstrap_ci is None else float(report.bootstrap_ci.total_return_p05)
        ),
        "n_positive_regimes": int(report.n_positive_regimes),
        "n_tested_regimes": int(n_tested),
        "holdout_total_return": (
            None if report.holdout is None else float(report.holdout.total_return)
        ),
        "provenance": provenance,
    }
    path = out_dir / "validation_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
```

In `validate()`, immediately after `write_tearsheet(...)`, add:

```python
validation_json = _write_validation_report_json(
    out_dir=out_dir,
    slug=strategy,
    run_date=date.today(),
    data_start=start_date,
    data_end=end_date,
    report=report,
    provenance=f"quant validate {strategy} --start {start_date} --end {end_date}",
)
```

After printing `Tear-sheet: ...`, add:

```python
console.print(f"Validation JSON: {validation_json}")
```

- [ ] **Step 6: Run refresh tests and targeted validation CLI smoke**

Run:

```bash
uv run pytest tests/governance/test_refresh.py tests/test_cli.py::test_validate_command_runs_to_completion_on_known_strategy -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add quant/governance/__init__.py quant/governance/refresh.py quant/cli.py tests/governance/test_refresh.py
git commit -m "feat(governance): build artifacts from validation evidence"
```

---

### Task 4: Governance CLI And Strategy Visibility

**Files:**
- Modify: `quant/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_cli.py`:

```python
def test_governance_help_succeeds(fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["governance", "--help"])
    assert result.exit_code == 0, result.output
    assert "status" in result.output
    assert "refresh" in result.output


def test_governance_refresh_writes_artifacts(tmp_data_dir: Path, fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["governance", "refresh", "--asof", "2026-05-26"])
    assert result.exit_code == 0, result.output
    assert (tmp_data_dir / "governance" / "validation_manifest.json").exists()
    assert (tmp_data_dir / "governance" / "strategy_states.json").exists()


def test_governance_status_renders_unknown_when_artifacts_missing(
    tmp_data_dir: Path, fake_env: None
) -> None:
    result = CliRunner().invoke(cli, ["governance", "status"])
    assert result.exit_code == 0, result.output
    assert "unknown" in result.output.lower()


def test_strategies_shows_governance_column(tmp_data_dir: Path, fake_env: None) -> None:
    result = CliRunner().invoke(cli, ["strategies"])
    assert result.exit_code == 0, result.output
    assert "Governance" in result.output
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run pytest tests/test_cli.py -v -k "governance or strategies_shows_governance"
```

Expected: FAIL because the command group and column do not exist.

- [ ] **Step 3: Add CLI governance helpers**

In `quant/cli.py`, add imports only inside commands where possible. Add this helper near `strategies()`:

```python
def _governance_state_labels(data_dir: Path) -> dict[str, str]:
    from quant.governance.models import GovernanceError
    from quant.governance.store import load_strategy_states, strategy_states_path

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError:
        return {}
    return {slug: state.state.value for slug, state in states.items()}
```

- [ ] **Step 4: Add `quant governance` group**

In `quant/cli.py`, before `data` group, add:

```python
@cli.group(help="Strategy governance and evidence-gated live eligibility.")
def governance() -> None:
    pass


@governance.command("refresh", help="Recompute governance artifacts from validation evidence.")
@click.option("--asof", default=None, help="Evaluation date (YYYY-MM-DD). Default: today.")
@click.option("--max-age-days", default=30, show_default=True, type=int)
def governance_refresh(asof: str | None, max_age_days: int) -> None:
    from quant.governance import GovernancePolicy, build_governance_artifacts

    settings = Settings()  # type: ignore[call-arg]
    asof_date = date.fromisoformat(asof) if asof else date.today()
    states = build_governance_artifacts(
        data_dir=settings.data_dir,
        registry=REGISTRY,
        policy=GovernancePolicy(max_validation_age_days=max_age_days),
        asof=asof_date,
    )
    table = Table(title=f"Governance refresh — {asof_date}", show_header=True)
    table.add_column("Strategy")
    table.add_column("State")
    table.add_column("Reason")
    for slug, state in sorted(states.items()):
        table.add_row(slug, state.state.value, state.reason)
    console.print(table)


@governance.command("status", help="Show current governance state for each strategy.")
def governance_status() -> None:
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import load_strategy_states, strategy_states_path

    settings = Settings()  # type: ignore[call-arg]
    try:
        states = load_strategy_states(strategy_states_path(settings.data_dir))
    except GovernanceError as exc:
        states = {}
        console.print(f"[yellow]{exc}; run `quant governance refresh`.[/yellow]")

    table = Table(title="Strategy governance", show_header=True)
    for col in ("Strategy", "Code Live", "Governance", "Age", "Reasons"):
        table.add_column(col)
    for spec in list_strategies():
        state = states.get(spec.slug)
        if state is None:
            table.add_row(
                spec.slug,
                "yes" if spec.enabled_live else "no",
                GovernanceState.UNKNOWN.value,
                "",
                "no governance artifact",
            )
            continue
        age = "" if state.validation_age_days is None else f"{state.validation_age_days}d"
        table.add_row(
            spec.slug,
            "yes" if spec.enabled_live else "no",
            state.state.value,
            age,
            ", ".join(state.reason_codes) or "ok",
        )
    console.print(table)
```

- [ ] **Step 5: Update `quant strategies` table**

In `strategies()`, add:

```python
settings = Settings.model_construct(data_dir=Path("./data")) if not _can_load_settings() else Settings()  # type: ignore[call-arg]
governance_labels = _governance_state_labels(settings.data_dir)
```

Add a column:

```python
table.add_column("Governance", justify="center")
```

Add one more row value:

```python
governance_labels.get(spec.slug, "unknown"),
```

- [ ] **Step 6: Run CLI tests**

Run:

```bash
uv run pytest tests/test_cli.py -v -k "governance or strategies_shows_governance"
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add quant/cli.py tests/test_cli.py
git commit -m "feat(cli): expose strategy governance status"
```

---

### Task 5: Rebalance Governance Filtering

**Files:**
- Modify: `quant/live/rebalance.py`
- Modify: `quant/cli.py`
- Modify: `tests/live/test_rebalance.py`

- [ ] **Step 1: Add failing rebalance tests**

Append to `tests/live/test_rebalance.py`:

```python
def _write_state_file(data_dir: Path, states: dict[str, str]) -> None:
    import json

    gov = data_dir / "governance"
    gov.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "strategies": {
            slug: {
                "slug": slug,
                "state": state,
                "evaluated_at": "2026-05-26T00:00:00",
                "validation_age_days": 1,
                "reason_codes": [] if state == "live" else ["failed_gate_regime"],
                "reason": "ok" if state == "live" else "blocked",
                "code_enabled_live": True,
                "manual_block": False,
            }
            for slug, state in states.items()
        },
    }
    (gov / "strategy_states.json").write_text(json.dumps(payload))


def test_default_rebalance_uses_only_governance_live_strategies(
    fake_settings: Settings, patched_bars: None
) -> None:
    _write_state_file(fake_settings.data_dir, {"momentum": "quarantined", "trend": "live"})
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == ["trend"]


def test_missing_governance_artifacts_fail_closed_for_default_rebalance(
    fake_settings: Settings, patched_bars: None
) -> None:
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == []
    assert report.skipped_reason is not None
    assert "governance" in report.skipped_reason.lower()


def test_include_quarantined_requires_dry_run(fake_settings: Settings, patched_bars: None) -> None:
    _write_state_file(fake_settings.data_dir, {"momentum": "quarantined"})
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=False,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        include_quarantined=True,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == []
    assert report.skipped_reason is not None
    assert "dry-run" in report.skipped_reason.lower()


def test_dry_run_can_include_quarantined_for_observation(
    fake_settings: Settings, patched_bars: None
) -> None:
    _write_state_file(fake_settings.data_dir, {"momentum": "quarantined"})
    client = _StubAlpacaClient()
    report = run_rebalance(
        asof=date(2024, 6, 28),
        dry_run=True,
        client=client,  # type: ignore[arg-type]
        settings=fake_settings,
        include_quarantined=True,
        skip_safety_checks=True,
    )
    assert report.enabled_strategies == ["momentum"]
```

- [ ] **Step 2: Run rebalance tests and verify they fail**

Run:

```bash
uv run pytest tests/live/test_rebalance.py -v -k "governance or quarantined"
```

Expected: FAIL because `run_rebalance()` lacks `include_quarantined` and governance filtering.

- [ ] **Step 3: Implement governance strategy selection**

In `quant/live/rebalance.py`, import governance types inside a new helper and add:

```python
def _governance_selected_strategies(
    data_dir: Path,
    *,
    include_quarantined: bool,
) -> tuple[list[str], str | None]:
    from quant.governance.models import GovernanceError, GovernanceState
    from quant.governance.store import load_strategy_states, strategy_states_path

    try:
        states = load_strategy_states(strategy_states_path(data_dir))
    except GovernanceError as exc:
        return [], f"governance unavailable: {exc}. Run `quant governance refresh`."

    selected: list[str] = []
    for slug, state in sorted(states.items()):
        if slug not in REGISTRY:
            continue
        if state.state is GovernanceState.LIVE:
            selected.append(slug)
        elif include_quarantined and state.state is GovernanceState.QUARANTINED:
            selected.append(slug)
    return selected, None
```

Change `run_rebalance` signature:

```python
    include_quarantined: bool = False,
) -> RebalanceReport:
```

After settings/client/asof setup and before account fetch, add:

```python
    if include_quarantined and not dry_run:
        return RebalanceReport(
            asof=asof,
            equity=0.0,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            skipped_reason="--include-quarantined is allowed only for dry-run observation.",
        )
```

Replace:

```python
enabled = strategies if strategies is not None else _enabled_strategies()
```

with:

```python
if strategies is not None:
    enabled = strategies
else:
    enabled, governance_error = _governance_selected_strategies(
        settings.data_dir,
        include_quarantined=include_quarantined,
    )
    if governance_error is not None:
        logger.error("{}", governance_error)
        return RebalanceReport(
            asof=asof,
            equity=account.equity,
            enabled_strategies=[],
            outcomes=[],
            dry_run=dry_run,
            safety_checks=safety_results,
            skipped_reason=governance_error,
        )
```

Keep explicit `strategies=[...]` as a test/backdoor path for targeted calls. CLI will filter normal user behavior.

- [ ] **Step 4: Wire CLI option**

In `quant/cli.py`, add to `rebalance` command options:

```python
@click.option(
    "--include-quarantined",
    is_flag=True,
    help="Dry-run only: include quarantined strategies for observation.",
)
```

Change signature:

```python
def rebalance(
    dry_run: bool,
    asof: str | None,
    strategy_filter: str | None,
    include_quarantined: bool,
) -> None:
```

Before calling `run_rebalance`, add:

```python
if include_quarantined and not dry_run:
    raise click.ClickException("--include-quarantined is allowed only with --dry-run.")
```

Pass the flag:

```python
report = run_rebalance(
    asof=asof_date,
    dry_run=dry_run,
    strategies=strategies_arg,
    include_quarantined=include_quarantined,
)
```

After printing the header, add:

```python
if report.skipped_reason:
    console.print(f"[yellow]Skipped: {report.skipped_reason}[/yellow]")
```

- [ ] **Step 5: Run rebalance tests**

Run:

```bash
uv run pytest tests/live/test_rebalance.py -v -k "governance or quarantined"
```

Expected: PASS.

- [ ] **Step 6: Run full live test file**

Run:

```bash
uv run pytest tests/live/test_rebalance.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add quant/live/rebalance.py quant/cli.py tests/live/test_rebalance.py
git commit -m "feat(live): gate rebalance through governance"
```

---

### Task 6: Documentation And Initial Conservative Artifacts

**Files:**
- Modify: `README.md`
- Create: `data/governance/validation_manifest.json`
- Create: `data/governance/strategy_states.json`

- [ ] **Step 1: Update README governance section**

In `README.md`, after the "Validation" section and before "Live paper trading", add:

```markdown
## Strategy governance

Normal paper rebalances are evidence-gated. `StrategySpec.enabled_live=True`
means a strategy is live-capable in code; governance decides whether it is
currently eligible for paper capital.

```bash
uv run quant validate trend
uv run quant governance refresh
uv run quant governance status
uv run quant rebalance --dry-run
```

`quant rebalance` fails closed when governance artifacts are missing or
malformed. Quarantined strategies can still be observed with:

```bash
uv run quant rebalance --dry-run --include-quarantined
```

`--include-quarantined` is rejected for non-dry-run rebalances.
```

- [ ] **Step 2: Generate initial governance artifacts**

Run:

```bash
uv run quant governance refresh --asof 2026-05-26
```

Expected: command succeeds and writes:

```text
data/governance/validation_manifest.json
data/governance/strategy_states.json
```

Because existing backtest directories do not yet contain `validation_report.json` sidecars until each strategy is revalidated, the initial conservative state may quarantine every live-capable strategy. That is acceptable fail-closed behavior.

- [ ] **Step 3: Inspect governance status**

Run:

```bash
uv run quant governance status
```

Expected: all registered strategies render with `live`, `quarantined`, `research`, or `unknown`; no traceback.

- [ ] **Step 4: Commit Task 6**

```bash
git add README.md data/governance/validation_manifest.json data/governance/strategy_states.json
git commit -m "docs(governance): document evidence-gated paper trading"
```

---

### Task 7: Final Verification

**Files:**
- No new files unless verification exposes failures.

- [ ] **Step 1: Run governance test suite**

Run:

```bash
uv run pytest tests/governance -v
```

Expected: PASS.

- [ ] **Step 2: Run CLI and rebalance tests**

Run:

```bash
uv run pytest tests/test_cli.py tests/live/test_rebalance.py -v
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 4: Run type and lint checks**

Run:

```bash
uv run ruff check .
uv run mypy quant
```

Expected: both PASS.

- [ ] **Step 5: Smoke CLI behavior**

Run:

```bash
uv run quant governance status
uv run quant strategies
uv run quant rebalance --dry-run
uv run quant rebalance --dry-run --include-quarantined
```

Expected:

- `governance status` renders a strategy table.
- `strategies` includes a Governance column.
- normal dry-run either trades only `live` strategies or fails closed with governance remediation.
- quarantined dry-run includes quarantined strategies for observation.

- [ ] **Step 6: Commit verification fixes if needed**

If verification required fixes, commit them:

```bash
git add quant tests README.md data/governance
git commit -m "fix(governance): pass final verification"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

**Spec coverage:** The plan implements operational states, deterministic JSON artifacts, governance refresh, CLI status, strategy visibility, fail-closed rebalance behavior, dry-run quarantined observation, and tests across policy/CLI/live paths.

**Placeholder scan:** No task depends on unspecified behavior. Manual live overrides are intentionally excluded from version one; manual blocks are represented on validation evidence.

**Type consistency:** The same names are used throughout: `GovernanceState`, `ValidationEvidence`, `StrategyState`, `GovernancePolicy`, `validation_manifest.json`, `strategy_states.json`, and `include_quarantined`.
