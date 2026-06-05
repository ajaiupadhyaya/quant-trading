# OrderTemplate Execution Fields (Raise-the-Ceiling Phase 1 foundation) — 2026-06-05

## Goal

Give `OrderTemplate` the data model for non-market execution (`order_type` /
`limit_price` / `time_in_force`) and have the submit path honor it — with
**defaults that reproduce today's market + DAY behavior byte-for-byte**. This is
the foundation the Phase 4 execution-quality work (LIMIT / TWAP / VWAP / POV) will
build on. **Provably zero live behavior change**: no live strategy ever emits a
non-MARKET template from this change.

Roadmap anchor: `docs/specs/2026-06-02-raise-the-ceiling-roadmap.md` Phase 1 —
"`OrderTemplate` gains `order_type`/`limit_price`/`time_in_force` — defaults
reproduce today's market+DAY byte-for-byte, COID unchanged, behind a per-strategy
flag (OFF)." The per-strategy flag / `ExecutionPolicy` / strategy wiring belong to
Phase 4 (human-gated) and are explicitly OUT of scope here.

## Constraints

- **Byte-identical default path.** A default `OrderTemplate` (MARKET / DAY /
  `limit_price=None`) must produce exactly the broker request today produces. A
  regression test pins this.
- **COID unchanged.** `make_client_order_id` and its call in `submit_order` are
  untouched.
- **SDK-free data model.** `orders.py` defines its own domain enums (like the
  existing `OrderSide`) so it imports no Alpaca SDK and stays trivially testable;
  the SDK mapping lives only in `alpaca.py`.
- **Existing construction sites untouched.** The new `OrderTemplate` fields carry
  defaults, so `netting.py`, `reconciler.py` (×3), and `winddown.py` keep
  constructing templates exactly as today.

## Architecture

### 1. `quant/execution/orders.py` — pure data model + validation

Add two domain enums (StrEnum, mirroring `OrderSide`):

```python
class OrderType(enum.StrEnum):
    MARKET = "market"
    LIMIT = "limit"

class TimeInForce(enum.StrEnum):
    DAY = "day"
    GTC = "gtc"
```

Extend the frozen `OrderTemplate` with three defaulted fields:

```python
@dataclass(frozen=True)
class OrderTemplate:
    symbol: str
    qty: int
    side: OrderSide
    strategy_slug: str
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    def __post_init__(self) -> None:
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None or not math.isfinite(self.limit_price) or self.limit_price <= 0:
                raise ValueError("LIMIT order requires a positive, finite limit_price")
        elif self.limit_price is not None:
            raise ValueError("MARKET order must not carry a limit_price")
```

(`math` import added.) Validation makes an invalid template impossible to
construct, so the submit path never has to defend against bad combinations.

### 2. `quant/execution/alpaca.py` — `submit_order` honors the template

Today it hardcodes `MarketOrderRequest(... time_in_force=TimeInForce.DAY)`. Change
to map the domain enums to the Alpaca SDK and branch on `order_type`:

- Map domain `TimeInForce` → Alpaca `TimeInForce` (`DAY`→`DAY`, `GTC`→`GTC`) via a
  small local dict.
- `MARKET` → build the **same** `MarketOrderRequest` as today: `symbol`, `qty`,
  `side`, mapped `time_in_force`, `client_order_id`. (For a default template this
  is byte-identical to the current code.)
- `LIMIT` → build a `LimitOrderRequest` (imported from `alpaca.trading.requests`)
  with the same fields plus `limit_price=order.limit_price`.
- Dry-run: the existing log line gains the order_type and, when LIMIT, the
  limit_price; still returns the COID and submits nothing.

The COID computation, the `asof` handling, and the `dry_run` early-return are
unchanged.

## Data flow

```
strategy / netting / reconciler / winddown
   └─ OrderTemplate(...)            # always defaults today: MARKET / DAY / None
        └─ submit_order(order, asof, dry_run)
             ├─ MARKET → MarketOrderRequest(... DAY ...)   # byte-identical to today
             └─ LIMIT  → LimitOrderRequest(... limit_price ...)   # unused by any live path
```

## Error handling

- Invalid field combinations raise `ValueError` at construction (`__post_init__`)
  — fail fast, before anything reaches the broker.
- Domain enums are closed sets; the TIF map covers both members. The submit
  branch is exhaustive over `OrderType` (MARKET vs LIMIT).
- No change to the existing submit failure / retry semantics.

## Testing (TDD — tests first)

`tests/execution/test_orders.py` (pure):
- A default `OrderTemplate` has `order_type=MARKET`, `time_in_force=DAY`,
  `limit_price=None`.
- `__post_init__` raises `ValueError` for LIMIT-without-price (None,
  non-positive, non-finite) and for MARKET-with-a-price.
- A LIMIT template with a positive price constructs successfully and round-trips
  its fields.

`tests/execution/test_alpaca.py` (or the existing submit test module — reuse its
fake/mock trading client that captures the submitted request):
- **Regression pin:** a default (market) template → `submit_order` calls the
  broker with a `MarketOrderRequest` whose `symbol`/`qty`/`side`/`time_in_force`
  (DAY)/`client_order_id` match today exactly.
- A LIMIT template → a `LimitOrderRequest` with the correct `limit_price` and TIF.
- `dry_run=True` returns the COID and the trading client is never called.
- GTC maps through correctly.

Full suite green (`uv run pytest`), `mypy quant/` strict clean, `ruff` clean —
including the untouched `netting`/`reconciler`/`winddown` tests, which prove the
construction sites are unaffected.

## Out of scope (Phase 4, human-gated)

`ExecutionPolicy`; any per-strategy flag/Settings wiring; routing a live strategy
to LIMIT; TWAP/VWAP/POV child-slicing; the intraday fill manager. This change adds
capability that nothing live exercises.

## Operational note

Implement on the `feat/order-exec-fields` worktree off `main`; the main checkout
is the live launchd host's tree and must stay on `main`. Order-path code, but with
byte-identical defaults + a regression pin and no live caller of the new fields,
so live order flow is unchanged.
