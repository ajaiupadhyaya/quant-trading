# OrderTemplate Execution Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `order_type`/`limit_price`/`time_in_force` to `OrderTemplate` and have `submit_order` honor them, with market+DAY defaults that reproduce today's broker request byte-for-byte.

**Architecture:** `orders.py` gains SDK-free domain enums (`OrderType`, `TimeInForce`) and three defaulted `OrderTemplate` fields with `__post_init__` validation. `alpaca.py`'s `submit_order` maps the domain enums to the Alpaca SDK and branches MARKET (byte-identical to today) vs LIMIT. No live caller ever sets non-default fields.

**Tech Stack:** Python 3.12, `alpaca-py` SDK, pytest (unittest.mock), `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-06-05-order-execution-fields-design.md`

**Working directory:** the `.worktrees/order-exec-fields` worktree on branch `feat/order-exec-fields` (the main checkout is the live launchd host's tree — do NOT switch it off `main`). Run all commands from the worktree root. Run `uv sync --all-extras` once before starting if the worktree's env is fresh.

---

## File Structure

- **Modify** `quant/execution/orders.py` — add `OrderType` + `TimeInForce` StrEnums; add three defaulted fields + `__post_init__` validation to `OrderTemplate`; add `import math`.
- **Modify** `quant/execution/alpaca.py` — `submit_order` honors `order.order_type`/`limit_price`/`time_in_force`; alias the SDK `TimeInForce` as `AlpacaTIF`, import `LimitOrderRequest`, map domain TIF → SDK.
- **Test** `tests/execution/test_orders.py` — pure data-model + validation tests.
- **Test** `tests/execution/test_alpaca.py` — submit_order regression-pin (market byte-identical) + LIMIT + TIF mapping; reuses the existing `mock_trading_client` fixture (`mock_trading_client.submit_order.call_args.args[0]` is the submitted request).

The four `OrderTemplate` construction sites (`netting.py`, `reconciler.py`, `winddown.py`) are NOT modified — the new fields' defaults keep them byte-identical.

---

### Task 1: `orders.py` — domain enums + validated `OrderTemplate` fields

**Files:**
- Modify: `quant/execution/orders.py`
- Test: `tests/execution/test_orders.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_orders.py`:

```python
import math

import pytest

from quant.execution.orders import (
    OrderSide,
    OrderTemplate,
    OrderType,
    TimeInForce,
)


def test_order_template_defaults_reproduce_today():
    t = OrderTemplate(symbol="SPY", qty=10, side=OrderSide.BUY, strategy_slug="momentum")
    assert t.order_type is OrderType.MARKET
    assert t.time_in_force is TimeInForce.DAY
    assert t.limit_price is None


def test_limit_template_requires_positive_price():
    t = OrderTemplate(
        symbol="SPY", qty=10, side=OrderSide.BUY, strategy_slug="momentum",
        order_type=OrderType.LIMIT, limit_price=420.5,
    )
    assert t.order_type is OrderType.LIMIT
    assert t.limit_price == 420.5


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("inf"), float("nan")])
def test_limit_without_valid_price_raises(bad):
    with pytest.raises(ValueError):
        OrderTemplate(
            symbol="SPY", qty=10, side=OrderSide.BUY, strategy_slug="momentum",
            order_type=OrderType.LIMIT, limit_price=bad,
        )


def test_market_with_a_limit_price_raises():
    with pytest.raises(ValueError):
        OrderTemplate(
            symbol="SPY", qty=10, side=OrderSide.BUY, strategy_slug="momentum",
            limit_price=420.5,  # MARKET default + a price is contradictory
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/execution/test_orders.py -q -p no:cacheprovider --no-cov`
Expected: FAIL — `ImportError: cannot import name 'OrderType'` (and `TimeInForce`).

- [ ] **Step 3: Implement**

In `quant/execution/orders.py`, add `import math` to the imports, add the two enums after `OrderSide`, and extend `OrderTemplate`:

```python
import math
```

```python
class OrderType(enum.StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(enum.StrEnum):
    DAY = "day"
    GTC = "gtc"
```

```python
@dataclass(frozen=True)
class OrderTemplate:
    """A target order to be submitted to Alpaca.

    `qty` is always a positive integer. `side` encodes direction. The execution
    fields default to MARKET / DAY / no-limit-price, reproducing the historical
    behavior byte-for-byte; non-default values are foundation for future
    execution-quality work and are not emitted by any live path today.
    """

    symbol: str
    qty: int
    side: OrderSide
    strategy_slug: str
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    def __post_init__(self) -> None:
        if self.order_type is OrderType.LIMIT:
            if (
                self.limit_price is None
                or not math.isfinite(self.limit_price)
                or self.limit_price <= 0
            ):
                raise ValueError("LIMIT order requires a positive, finite limit_price")
        elif self.limit_price is not None:
            raise ValueError("MARKET order must not carry a limit_price")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/execution/test_orders.py -q -p no:cacheprovider --no-cov`
Expected: PASS (the new tests plus any pre-existing ones in the file).

- [ ] **Step 5: Verify construction sites still work**

Run: `uv run pytest tests/execution/test_netting.py tests/execution/test_reconciler.py -q -p no:cacheprovider --no-cov`
Expected: PASS — proves `netting.py`/`reconciler.py` constructions are unaffected by the new defaulted fields.

- [ ] **Step 6: Commit**

```bash
git add quant/execution/orders.py tests/execution/test_orders.py
git commit -m "feat(execution): add validated execution fields to OrderTemplate (market+DAY defaults)"
```

---

### Task 2: `alpaca.py` — `submit_order` honors the template

**Files:**
- Modify: `quant/execution/alpaca.py` (imports near lines 14-17; `submit_order` lines ~154-184)
- Test: `tests/execution/test_alpaca.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_alpaca.py`:

```python
def test_submit_order_market_request_is_byte_identical(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    """A default (market) template must submit the same MarketOrderRequest as before."""
    from datetime import date

    from alpaca.trading.enums import OrderSide as AlpacaSide
    from alpaca.trading.enums import TimeInForce as AlpacaTIF
    from alpaca.trading.requests import MarketOrderRequest

    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        client.submit_order(
            OrderTemplate(symbol="SPY", qty=3, side=OrderSide.BUY, strategy_slug="trend"),
            asof=date(2026, 6, 1),
        )
    req = mock_trading_client.submit_order.call_args.args[0]
    assert isinstance(req, MarketOrderRequest)
    assert req.symbol == "SPY"
    assert req.qty == 3
    assert req.side == AlpacaSide.BUY
    assert req.time_in_force == AlpacaTIF.DAY
    assert req.client_order_id == "trend-20260601-SPY"


def test_submit_order_limit_builds_limit_request(
    fake_env: None, mock_trading_client: MagicMock
) -> None:
    from datetime import date

    from alpaca.trading.enums import TimeInForce as AlpacaTIF
    from alpaca.trading.requests import LimitOrderRequest

    from quant.execution.orders import OrderType, TimeInForce

    with patch("quant.execution.alpaca.TradingClient", return_value=mock_trading_client):
        client = AlpacaClient()
        client.submit_order(
            OrderTemplate(
                symbol="SPY", qty=2, side=OrderSide.SELL, strategy_slug="trend",
                order_type=OrderType.LIMIT, limit_price=415.25, time_in_force=TimeInForce.GTC,
            ),
            asof=date(2026, 6, 1),
        )
    req = mock_trading_client.submit_order.call_args.args[0]
    assert isinstance(req, LimitOrderRequest)
    assert req.limit_price == 415.25
    assert req.time_in_force == AlpacaTIF.GTC
    assert req.client_order_id == "trend-20260601-SPY"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/execution/test_alpaca.py -q -p no:cacheprovider --no-cov -k "byte_identical or limit_builds"`
Expected: FAIL — today every submission is a `MarketOrderRequest` (the LIMIT test fails the `isinstance(LimitOrderRequest)` assertion).

- [ ] **Step 3: Implement**

In `quant/execution/alpaca.py`, update the imports (lines 14-17) so the SDK `TimeInForce` is aliased (avoids colliding with the domain enum) and `LimitOrderRequest` is available:

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
```

Add the domain `TimeInForce`/`OrderType` to the existing orders import at line 19 (keep `make_client_order_id`):

```python
from quant.execution.orders import (
    OrderSide,
    OrderTemplate,
    OrderType,
    TimeInForce,
    make_client_order_id,
)
```

Add a module-level TIF map (near the top, after imports):

```python
_TIF_MAP = {TimeInForce.DAY: AlpacaTIF.DAY, TimeInForce.GTC: AlpacaTIF.GTC}
```

Replace the request-construction block inside `submit_order` (the `req = MarketOrderRequest(...)` assignment, lines ~166-172) with a branch on `order_type`:

```python
        coid = make_client_order_id(order.strategy_slug, order.symbol, asof or date.today())
        side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL
        tif = _TIF_MAP[order.time_in_force]
        if order.order_type is OrderType.LIMIT:
            req: MarketOrderRequest | LimitOrderRequest = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=coid,
                limit_price=order.limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=coid,
            )
```

Update the dry-run log line to include the order type (and limit price when set) — replace the existing `logger.info("[DRY-RUN] would submit ...")` block:

```python
        if dry_run:
            logger.info(
                "[DRY-RUN] would submit {} {} {} {}{} (coid={})",
                order.side,
                order.qty,
                order.symbol,
                order.order_type,
                f"@{order.limit_price}" if order.limit_price is not None else "",
                coid,
            )
            return coid
```

The `self._trading.submit_order(req)` call and success log remain unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/execution/test_alpaca.py -q -p no:cacheprovider --no-cov`
Expected: PASS — including the four pre-existing submit tests (byte-identical default path preserved).

- [ ] **Step 5: Commit**

```bash
git add quant/execution/alpaca.py tests/execution/test_alpaca.py
git commit -m "feat(execution): submit_order honors OrderTemplate exec fields (market path byte-identical)"
```

---

### Task 3: Full verification (suite + mypy + ruff)

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: PASS. (If any `tests/intraday/data/*` or `tests/strategies/test_multi_factor.py::test_fundamentals_panel_uses_real_market_cap_not_price` fail, re-run them in isolation — they are flaky network/data-dependent tests unrelated to this change and pass on their own; CI on `main` is green.)

- [ ] **Step 2: mypy strict (CI scope)**

Run: `uv run mypy quant/`
Expected: `Success: no issues found`.

- [ ] **Step 3: ruff check + format**

Run: `uv run ruff check quant/execution/ tests/execution/ && uv run ruff format --check quant/execution/orders.py quant/execution/alpaca.py`
Expected: clean. (Run `uv run ruff format <files>` then re-commit if format flags anything.)

- [ ] **Step 4: Commit any format fix**

```bash
git add -A && git commit -m "chore(execution): ruff format"
```

(Skip if nothing to format.)

---

## Self-Review Notes

- **Spec coverage:** domain enums + `OrderTemplate` fields + validation (Task 1) ✓; `submit_order` honors fields, market byte-identical + LIMIT (Task 2) ✓; construction sites untouched, verified (Task 1 Step 5 + full suite) ✓; mypy/ruff/suite (Task 3) ✓. Out-of-scope (ExecutionPolicy, strategy wiring) correctly absent.
- **Type consistency:** domain `OrderType`/`TimeInForce` and field names (`order_type`, `limit_price`, `time_in_force`) are identical across `orders.py`, the `_TIF_MAP`, `submit_order`, and all tests. The SDK `TimeInForce` is consistently aliased `AlpacaTIF` to avoid the name collision with the domain enum.
- **Placeholder scan:** none — every code step shows complete code; the only "find the line" reference (the existing `quant.execution.orders` import in `alpaca.py`) names the exact symbols to add.
- **Byte-identical guarantee:** Task 2 Step 1's regression test pins the default (market) request's symbol/qty/side/TIF(DAY)/coid; the four pre-existing submit tests must still pass.
