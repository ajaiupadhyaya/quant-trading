"""CI smoke: drive `run_rebalance` against synthetic bars + a stubbed Alpaca.

The dry-run path inside ``quant.live.rebalance.run_rebalance`` never touches the
network when ``client`` is injected, so this script can run in CI without
Alpaca credentials. We monkey-patch ``_bars_for`` to return synthetic data
keyed on the strategy's universe, then assert every strategy outcome was
error-free.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from quant.execution.alpaca import AccountInfo
from quant.live.rebalance import run_rebalance
from tests.conftest import synthetic_bars


class _StubAlpaca:
    def account(self) -> AccountInfo:
        return AccountInfo(
            equity=100_000.0,
            last_equity=100_000.0,
            buying_power=200_000.0,
            cash=10_000.0,
            portfolio_value=100_000.0,
            pattern_day_trader=False,
        )

    def positions(self) -> list:  # type: ignore[type-arg]
        return []

    def submit_order(self, order, *, dry_run: bool = False) -> str:  # type: ignore[no-untyped-def]
        return f"stub-{order.strategy_slug}-{order.symbol}"


def _stub_bars(strategy_cls, asof, history_days):  # type: ignore[no-untyped-def]
    return synthetic_bars(
        list(strategy_cls.spec.universe),
        date(asof.year - 3, asof.month, asof.day),
        asof,
        seed=7,
    )


def main() -> int:
    with patch("quant.live.rebalance._bars_for", _stub_bars):
        report = run_rebalance(
            asof=date.today(),
            dry_run=True,
            client=_StubAlpaca(),  # type: ignore[arg-type]
        )
    print(f"smoke: {len(report.outcomes)} outcomes; total_orders={report.total_orders}")
    errors = [(o.slug, o.error) for o in report.outcomes if o.error]
    if errors:
        print(f"ERRORS: {errors}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
