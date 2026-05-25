"""Hypothesis property tests for the reconciler.

The reconciler is one of the few places where a subtle off-by-one would silently
desync our per-strategy bookkeeping from Alpaca. The example tests in
test_reconciler.py cover a handful of cases; these property tests check the
invariants that must hold for *any* (current, target) state pair:

  * If target == current, no orders are emitted.
  * Every order has a strictly positive qty (the sign lives in the side).
  * Sum of signed deltas across emitted orders equals target - current per symbol.
  * A long→short or short→long flip emits exactly two orders for that symbol.
  * The set of symbols in the output is exactly the set of symbols where
    target[sym] != current.get(sym, 0).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quant.execution.orders import OrderSide
from quant.execution.reconciler import reconcile

SYMBOL = st.sampled_from(["SPY", "TLT", "GLD", "IEF", "VNQ", "EFA", "EEM", "DBC"])
QTY = st.integers(min_value=-1_000, max_value=1_000)

POSITIONS = st.dictionaries(SYMBOL, QTY, min_size=0, max_size=8)


def _signed(order_side: OrderSide, qty: int) -> int:
    return qty if order_side is OrderSide.BUY else -qty


@given(current=POSITIONS)
@settings(max_examples=200, deadline=None)
def test_identity_reconciliation_emits_nothing(current: dict[str, int]) -> None:
    assert reconcile(target=current, current=current, strategy_slug="x") == []


@given(target=POSITIONS, current=POSITIONS)
@settings(max_examples=300, deadline=None)
def test_orders_have_positive_qty_and_correct_slug(
    target: dict[str, int], current: dict[str, int]
) -> None:
    orders = reconcile(target=target, current=current, strategy_slug="my-strat")
    for o in orders:
        assert o.qty > 0
        assert o.strategy_slug == "my-strat"


@given(target=POSITIONS, current=POSITIONS)
@settings(max_examples=300, deadline=None)
def test_signed_deltas_reconcile_to_target(target: dict[str, int], current: dict[str, int]) -> None:
    """Sum of signed order qtys per symbol equals target - current."""
    orders = reconcile(target=target, current=current, strategy_slug="x")
    per_sym: dict[str, int] = {}
    for o in orders:
        per_sym[o.symbol] = per_sym.get(o.symbol, 0) + _signed(o.side, o.qty)
    all_syms = set(target) | set(current)
    for sym in all_syms:
        expected = target.get(sym, 0) - current.get(sym, 0)
        assert per_sym.get(sym, 0) == expected


@given(target=POSITIONS, current=POSITIONS)
@settings(max_examples=300, deadline=None)
def test_zero_crossings_emit_two_orders(target: dict[str, int], current: dict[str, int]) -> None:
    """Long→short or short→long flips produce exactly 2 orders for that symbol."""
    orders = reconcile(target=target, current=current, strategy_slug="x")
    per_sym_count: dict[str, int] = {}
    for o in orders:
        per_sym_count[o.symbol] = per_sym_count.get(o.symbol, 0) + 1
    for sym in set(target) | set(current):
        cur = current.get(sym, 0)
        tgt = target.get(sym, 0)
        is_crossing = (cur > 0 and tgt < 0) or (cur < 0 and tgt > 0)
        if is_crossing:
            assert per_sym_count.get(sym, 0) == 2, (
                f"expected 2 orders for crossing {sym}: cur={cur}, tgt={tgt}"
            )


@given(target=POSITIONS, current=POSITIONS)
@settings(max_examples=300, deadline=None)
def test_emitted_symbol_set_matches_diff_set(
    target: dict[str, int], current: dict[str, int]
) -> None:
    orders = reconcile(target=target, current=current, strategy_slug="x")
    emitted = {o.symbol for o in orders}
    expected = {
        sym for sym in (set(target) | set(current)) if target.get(sym, 0) != current.get(sym, 0)
    }
    assert emitted == expected
