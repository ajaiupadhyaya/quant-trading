"""Portfolio-level pre-trade risk checks."""

from __future__ import annotations

from dataclasses import dataclass

from quant.execution.orders import OrderSide, OrderTemplate


@dataclass(frozen=True)
class RiskLimits:
    max_gross_exposure: float = 1.0
    max_symbol_weight: float = 0.40


@dataclass(frozen=True)
class RiskViolation:
    code: str
    detail: str
    symbol: str | None = None


@dataclass(frozen=True)
class PretradeReport:
    equity: float
    gross_exposure: float
    symbol_weights: dict[str, float]
    violations: list[RiskViolation]

    @property
    def passed(self) -> bool:
        return not self.violations


def build_pretrade_report(
    *,
    equity: float,
    orders: list[OrderTemplate],
    reference_prices: dict[str, float],
    limits: RiskLimits | None = None,
) -> PretradeReport:
    limits = limits or RiskLimits()
    symbol_notional: dict[str, float] = {}
    for order in orders:
        price = float(reference_prices.get(order.symbol, 0.0))
        signed = 1.0 if order.side is OrderSide.BUY else -1.0
        symbol_notional[order.symbol] = (
            symbol_notional.get(order.symbol, 0.0) + signed * order.qty * price
        )

    denominator = max(float(equity), 1e-9)
    symbol_weights = {
        symbol: abs(notional) / denominator for symbol, notional in sorted(symbol_notional.items())
    }
    gross = round(sum(symbol_weights.values()), 12)
    violations: list[RiskViolation] = []
    if gross > limits.max_gross_exposure:
        violations.append(
            RiskViolation(
                code="gross_exposure",
                detail=f"gross exposure {gross:.2%} exceeds {limits.max_gross_exposure:.2%}",
            )
        )
    for symbol, weight in symbol_weights.items():
        if weight > limits.max_symbol_weight:
            violations.append(
                RiskViolation(
                    code="symbol_concentration",
                    detail=f"{symbol} weight {weight:.2%} exceeds {limits.max_symbol_weight:.2%}",
                    symbol=symbol,
                )
            )
    return PretradeReport(
        equity=float(equity),
        gross_exposure=gross,
        symbol_weights=symbol_weights,
        violations=violations,
    )
