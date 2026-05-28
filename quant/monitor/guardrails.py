"""Pure guardrail evaluation. No I/O, no side effects, total functions.

Each guardrail inspects one aspect of book health and yields a
``GuardrailOutcome`` with severity in {ok, warn, halt}. The overall tick halts
iff any guardrail returns ``halt``. Halt authority belongs to drift and
account-drawdown (computed from authoritative local equity history);
reconciliation and bar-freshness are warn-only by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from quant.governance.drift import DriftConfig, DriftRow
from quant.live.safety import CheckResult, StrategyRiskBudget

Severity = Literal["ok", "warn", "halt"]
_RANK: dict[Severity, int] = {"ok": 0, "warn": 1, "halt": 2}


@dataclass(frozen=True)
class GuardrailOutcome:
    name: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class GuardrailConfig:
    drift: DriftConfig = field(default_factory=DriftConfig)
    risk: StrategyRiskBudget = field(default_factory=StrategyRiskBudget)
    reconciliation_is_halt: bool = False


@dataclass(frozen=True)
class GuardrailInputs:
    drift_rows: list[DriftRow]
    account_drawdown_pct: float  # non-positive
    latest_equity: float
    reconciliation: CheckResult | None  # None => skipped (no live account)
    bar_freshness: CheckResult | None  # None => skipped


@dataclass(frozen=True)
class GuardrailReport:
    outcomes: list[GuardrailOutcome]

    @property
    def worst_severity(self) -> Severity:
        worst: Severity = "ok"
        for o in self.outcomes:
            if _RANK[o.severity] > _RANK[worst]:
                worst = o.severity
        return worst

    @property
    def halting(self) -> bool:
        return self.worst_severity == "halt"


def evaluate_drift(rows: list[DriftRow]) -> GuardrailOutcome:
    halts = [r for r in rows if r.flag == "halt_candidate"]
    if halts:
        which = ", ".join(f"{r.strategy}@{r.window}d z={r.z_score:.2f}" for r in halts[:5])
        return GuardrailOutcome("drift", "halt", f"halt_candidate: {which}")
    watches = [r for r in rows if r.flag == "watch"]
    if watches:
        which = ", ".join(f"{r.strategy}@{r.window}d z={r.z_score:.2f}" for r in watches[:5])
        return GuardrailOutcome("drift", "warn", f"watch: {which}")
    if not rows:
        return GuardrailOutcome("drift", "ok", "no drift history")
    return GuardrailOutcome("drift", "ok", "all windows normal")


def evaluate_account_drawdown(dd_pct: float, budget: StrategyRiskBudget) -> GuardrailOutcome:
    cap = abs(budget.max_drawdown)
    if dd_pct <= -cap:
        return GuardrailOutcome("account_drawdown", "halt", f"drawdown {dd_pct:.2%} <= -{cap:.2%}")
    return GuardrailOutcome("account_drawdown", "ok", f"drawdown {dd_pct:.2%} within -{cap:.2%}")


def evaluate_reconciliation(recon: CheckResult | None, *, halt_on_breach: bool) -> GuardrailOutcome:
    if recon is None:
        return GuardrailOutcome("reconciliation", "ok", "skipped: no account")
    if recon.ok:
        return GuardrailOutcome("reconciliation", "ok", recon.detail)
    severity: Severity = "halt" if halt_on_breach else "warn"
    return GuardrailOutcome("reconciliation", severity, recon.detail)


def evaluate_bar_freshness(freshness: CheckResult | None) -> GuardrailOutcome:
    if freshness is None:
        return GuardrailOutcome("bar_freshness", "ok", "skipped")
    if freshness.ok:
        return GuardrailOutcome("bar_freshness", "ok", freshness.detail)
    return GuardrailOutcome("bar_freshness", "warn", freshness.detail)


def evaluate_guardrails(inputs: GuardrailInputs, config: GuardrailConfig) -> GuardrailReport:
    outcomes = [
        evaluate_drift(inputs.drift_rows),
        evaluate_account_drawdown(inputs.account_drawdown_pct, config.risk),
        evaluate_reconciliation(
            inputs.reconciliation, halt_on_breach=config.reconciliation_is_halt
        ),
        evaluate_bar_freshness(inputs.bar_freshness),
    ]
    return GuardrailReport(outcomes=outcomes)
