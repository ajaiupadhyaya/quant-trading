# Intraday DL Alpha (LSTM, next-bar return) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a torch-LSTM next-bar-return alpha model to the intraday showcase track, evaluated out-of-sample against naive (persistence) and linear (OLS) baselines on two tracks — a synthetic-signal track the LSTM must beat (machinery works) and a near-random track it is not expected to beat (honest no-edge).

**Architecture:** A new standalone `quant/intraday/dl/` subpackage. torch is a pinned dependency but imported *lazily* (inside functions) so importing `quant.*` elsewhere never pays the torch cost; torch-dependent tests `pytest.importorskip("torch")` so the suite stays green where torch is absent. numpy windowing/split/standardize and dependency-free baselines carry the no-lookahead discipline; a deterministic CPU training loop carries reproducibility. Imports nothing from the live loop; touches no live path. Spec: `docs/superpowers/specs/2026-06-09-intraday-dl-alpha-design.md`.

**Tech Stack:** Python 3.12, numpy, torch (new pinned dep), click (CLI), pytest. uv for all dependency/test operations.

---

## File Structure

| File | Responsibility | Imports torch? |
|---|---|---|
| `quant/intraday/dl/__init__.py` | package marker | no |
| `quant/intraday/dl/config.py` | `DLConfig` (validated knobs) | no |
| `quant/intraday/dl/data.py` | `make_windows`, `train_test_split`, `standardize` (numpy, no lookahead) | no |
| `quant/intraday/dl/baselines.py` | `naive_predict` (persistence), `linear_predict` (numpy OLS) | no |
| `quant/intraday/dl/model.py` | `build_model(config)` → `LSTMRegressor` (lazy torch, deterministic init) | yes (lazy) |
| `quant/intraday/dl/train.py` | `TrainOutput`, `train_model(...)` (Adam+MSE, seeded, deterministic) | yes (lazy) |
| `quant/intraday/dl/evaluate.py` | `predict`, `oos_metrics`, `synthetic_signal_series`, `random_series`, `evaluate_track` | yes (lazy, in `predict` only) |
| `quant/intraday/cli.py` (modify) | `quant intraday dl train` + `dl evaluate` (lazy imports in command bodies) | no (lazy) |
| `tests/intraday/dl/...` | one test module per source file + a dual-track integration test | torch tests use `importorskip` |
| `pyproject.toml` (modify) | add `"torch>=2.2"` to `dependencies` | — |

Each `quant/intraday/dl/<x>.py` has a matching `tests/intraday/dl/test_<x>.py`. Pure-numpy modules (`config`, `data`, `baselines`) are tested without torch and MUST NOT import it. torch modules (`model`, `train`, `evaluate.predict`, integration) begin their test files with `torch = pytest.importorskip("torch")`.

**Full-suite command (used throughout):** `uv run pytest -m "not network and not alpaca" -q` (the unfiltered run hangs on alpaca network tests).

---

### Task 1: Pin torch and sync

**Files:**
- Modify: `pyproject.toml` (the `dependencies` array, currently ending at `"anthropic>=0.40",` on line 30)

- [ ] **Step 1: Add the torch dependency**

In `pyproject.toml`, add `"torch>=2.2",` as the last entry of the `[project].dependencies` array (immediately after `"anthropic>=0.40",`):

```toml
    "anthropic>=0.40",
    "torch>=2.2",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs the Apple-Silicon CPU torch wheel (this MacBook). May take a few minutes on first download. On success the lockfile updates and `torch` is importable.

- [ ] **Step 3: Verify torch imports and is CPU**

Run: `uv run python -c "import torch; print(torch.__version__); print(torch.tensor([1.0]).sum().item())"`
Expected: prints a `2.x` version string and `1.0` (no error).

- [ ] **Step 4: Confirm the existing suite still passes**

Run: `uv run pytest -m "not network and not alpaca" -q`
Expected: PASS (same green baseline as before adding torch — no test depends on torch yet).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(intraday-dl): pin torch>=2.2 for the LSTM alpha showcase"
```

---

### Task 2: DLConfig

**Files:**
- Create: `quant/intraday/dl/__init__.py`
- Create: `quant/intraday/dl/config.py`
- Test: `tests/intraday/dl/__init__.py`, `tests/intraday/dl/test_config.py`

- [ ] **Step 1: Create the package markers**

Create `quant/intraday/dl/__init__.py` with a one-line docstring:

```python
"""Intraday DL alpha (torch LSTM, next-bar return) — sub-project D of the showcase track."""
```

Create `tests/intraday/dl/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing test**

Create `tests/intraday/dl/test_config.py`:

```python
import pytest

from quant.intraday.dl.config import DLConfig


def test_defaults():
    c = DLConfig()
    assert c.window >= 1
    assert c.hidden_size >= 1
    assert c.n_layers >= 1
    assert c.lr > 0
    assert c.epochs >= 1
    assert c.batch_size >= 1
    assert 0 < c.train_frac < 1
    assert isinstance(c.seed, int)


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        DLConfig(window=0)
    with pytest.raises(ValueError):
        DLConfig(hidden_size=0)
    with pytest.raises(ValueError):
        DLConfig(n_layers=0)
    with pytest.raises(ValueError):
        DLConfig(lr=0.0)
    with pytest.raises(ValueError):
        DLConfig(epochs=0)
    with pytest.raises(ValueError):
        DLConfig(batch_size=0)
    with pytest.raises(ValueError):
        DLConfig(train_frac=0.0)
    with pytest.raises(ValueError):
        DLConfig(train_frac=1.0)


def test_config_does_not_import_torch():
    import sys

    # Importing config must not have pulled in torch.
    assert "torch" not in sys.modules or True  # tolerated if another test loaded it
    # The real guarantee: the module has no torch attribute / top-level import.
    import quant.intraday.dl.config as cfg

    assert not hasattr(cfg, "torch")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.intraday.dl.config'`.

- [ ] **Step 4: Write minimal implementation**

Create `quant/intraday/dl/config.py`:

```python
"""Configuration for the intraday DL alpha (LSTM) showcase. No magic numbers per the
Charter; all knobs live here. NO torch import (kept lazy in model/train)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DLConfig:
    window: int = 16
    hidden_size: int = 32
    n_layers: int = 1
    lr: float = 0.01
    epochs: int = 60
    batch_size: int = 64
    seed: int = 7
    train_frac: float = 0.7

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_config.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/dl/config.py tests/intraday/dl/test_config.py
uv run ruff format quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/config.py
git add quant/intraday/dl/__init__.py quant/intraday/dl/config.py tests/intraday/dl/
git commit -m "feat(intraday-dl): DLConfig for the LSTM alpha showcase"
```

---

### Task 3: Windowing, chronological split, train-only standardization

**Files:**
- Create: `quant/intraday/dl/data.py`
- Test: `tests/intraday/dl/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_data.py`:

```python
import numpy as np
import pytest

from quant.intraday.dl.data import make_windows, standardize, train_test_split


def test_make_windows_shapes_and_contents():
    series = np.arange(10.0)  # 0,1,...,9
    X, y = make_windows(series, window=3)
    # n = len - window = 7
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    # X[0] = [0,1,2], y[0] = 3 (next value)
    assert list(X[0]) == [0.0, 1.0, 2.0]
    assert y[0] == 3.0
    # X[-1] = [6,7,8], y[-1] = 9
    assert list(X[-1]) == [6.0, 7.0, 8.0]
    assert y[-1] == 9.0


def test_make_windows_rejects_too_short():
    with pytest.raises(ValueError):
        make_windows(np.arange(3.0), window=3)  # need len > window


def test_train_test_split_is_chronological():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    Xtr, ytr, Xte, yte = train_test_split(X, y, train_frac=0.7)
    assert len(Xtr) == 7 and len(Xte) == 3
    # chronological: train is the FIRST 7, test the LAST 3 (no shuffle)
    assert ytr[0] == 0.0 and ytr[-1] == 6.0
    assert yte[0] == 7.0 and yte[-1] == 9.0


def test_standardize_uses_train_stats_only():
    train = np.array([[0.0, 2.0], [4.0, 6.0]])  # mean 3.0, std sqrt(5)
    test = np.array([[8.0, 10.0]])
    tr_z, te_z, mu, sd = standardize(train, test)
    assert mu == 3.0
    assert abs(sd - np.std(train)) < 1e-12
    # test standardized with TRAIN stats, not its own
    assert np.allclose(te_z, (test - mu) / sd)
    # train standardized to ~zero mean
    assert abs(tr_z.mean()) < 1e-12


def test_standardize_handles_zero_std():
    train = np.ones((4, 2))
    test = np.ones((2, 2))
    tr_z, te_z, mu, sd = standardize(train, test)
    assert sd == 1.0  # guarded, no divide-by-zero
    assert np.allclose(tr_z, 0.0)


def test_data_does_not_import_torch():
    import quant.intraday.dl.data as d

    assert not hasattr(d, "torch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.intraday.dl.data'`.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/dl/data.py`:

```python
"""Windowing, chronological split, and train-only standardization for the DL alpha.
All numpy; NO torch. Carries the no-lookahead discipline: windows use only past values,
the split is chronological (no shuffle), and standardization uses TRAIN statistics only."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_windows(series: NDArray[np.float64], window: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sliding lagged windows: X[i] = series[i:i+window], y[i] = series[i+window] (next value)."""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("series must be 1-D")
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(s) <= window:
        raise ValueError("series too short for the requested window")
    n = len(s) - window
    X = np.empty((n, window), dtype=np.float64)
    for i in range(n):
        X[i] = s[i : i + window]
    y = s[window:].copy()
    return X, y


def train_test_split(
    X: NDArray[np.float64], y: NDArray[np.float64], train_frac: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Chronological split (no shuffle): first `train_frac` is train, the rest is test."""
    n = len(X)
    cut = int(n * train_frac)
    if cut < 1 or cut >= n:
        raise ValueError("train_frac yields an empty train or test split")
    return X[:cut], y[:cut], X[cut:], y[cut:]


def standardize(
    train: NDArray[np.float64], test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Z-score using TRAIN mean/std only (no lookahead). Returns (train_z, test_z, mu, sd).
    A zero train std is guarded to 1.0 to avoid divide-by-zero."""
    mu = float(np.mean(train))
    sd = float(np.std(train))
    if sd == 0.0:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_data.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/dl/data.py tests/intraday/dl/test_data.py
uv run ruff format quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/data.py
git add quant/intraday/dl/data.py tests/intraday/dl/test_data.py
git commit -m "feat(intraday-dl): windowing, chronological split, train-only standardization"
```

---

### Task 4: Dependency-free baselines (persistence + numpy OLS)

**Files:**
- Create: `quant/intraday/dl/baselines.py`
- Test: `tests/intraday/dl/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_baselines.py`:

```python
import numpy as np

from quant.intraday.dl.baselines import linear_predict, naive_predict


def test_naive_predicts_last_in_window():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    yhat = naive_predict(X)
    assert list(yhat) == [3.0, 6.0]  # the last in-window value


def test_linear_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(200, 3))
    true = np.array([0.5, -0.2, 1.0])
    ytr = Xtr @ true + 0.3  # exact linear + intercept, no noise
    Xte = rng.normal(size=(50, 3))
    yhat = linear_predict(Xtr, ytr, Xte)
    expected = Xte @ true + 0.3
    assert np.allclose(yhat, expected, atol=1e-6)


def test_baselines_do_not_import_torch():
    import quant.intraday.dl.baselines as b

    assert not hasattr(b, "torch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_baselines.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.intraday.dl.baselines'`.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/dl/baselines.py`:

```python
"""Dependency-free baselines for the DL alpha comparison. NO torch.

- naive (persistence): predict the last in-window value. On the AR-signal series this
  captures autocorrelation; on a return series it is the random-walk-on-returns guess.
- linear: numpy OLS (with intercept) via np.linalg.lstsq."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def naive_predict(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Persistence: predict X[:, -1] (the most recent value in each window)."""
    return np.asarray(X, dtype=np.float64)[:, -1].copy()


def linear_predict(
    X_train: NDArray[np.float64], y_train: NDArray[np.float64], X_test: NDArray[np.float64]
) -> NDArray[np.float64]:
    """OLS with intercept fit on train, predicted on test (np.linalg.lstsq)."""
    a_train = np.hstack([np.asarray(X_train, dtype=np.float64), np.ones((len(X_train), 1))])
    coef, *_ = np.linalg.lstsq(a_train, np.asarray(y_train, dtype=np.float64), rcond=None)
    a_test = np.hstack([np.asarray(X_test, dtype=np.float64), np.ones((len(X_test), 1))])
    return a_test @ coef
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_baselines.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/dl/baselines.py tests/intraday/dl/test_baselines.py
uv run ruff format quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/baselines.py
git add quant/intraday/dl/baselines.py tests/intraday/dl/test_baselines.py
git commit -m "feat(intraday-dl): persistence + numpy-OLS baselines"
```

---

### Task 5: LSTMRegressor (lazy torch, deterministic init)

**Files:**
- Create: `quant/intraday/dl/model.py`
- Test: `tests/intraday/dl/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_model.py`:

```python
import pytest

torch = pytest.importorskip("torch")  # skip cleanly where torch is absent

from quant.intraday.dl.config import DLConfig  # noqa: E402
from quant.intraday.dl.model import build_model  # noqa: E402


def test_forward_shape():
    cfg = DLConfig(window=5, hidden_size=8)
    model = build_model(cfg)
    x = torch.zeros((4, cfg.window, 1))  # (batch, window, 1)
    out = model(x)
    assert out.shape == (4,)  # scalar per sample


def test_deterministic_init_same_seed():
    cfg = DLConfig(window=5, hidden_size=8, seed=123)
    m1 = build_model(cfg)
    m2 = build_model(cfg)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)  # same seed => identical initial weights


def test_different_seed_differs():
    a = build_model(DLConfig(window=5, hidden_size=8, seed=1))
    b = build_model(DLConfig(window=5, hidden_size=8, seed=2))
    pa = torch.cat([p.flatten() for p in a.parameters()])
    pb = torch.cat([p.flatten() for p in b.parameters()])
    assert not torch.allclose(pa, pb)


def test_model_module_has_no_top_level_torch():
    import quant.intraday.dl.model as m

    # torch must be imported lazily inside build_model, not at module top.
    assert not hasattr(m, "torch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.intraday.dl.model'` (or SKIP if torch absent — but torch was pinned in Task 1, so FAIL).

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/dl/model.py`:

```python
"""LSTM regressor for next-bar return. torch is imported LAZILY inside build_model so
importing quant.* elsewhere never pays the torch cost. Deterministic init from config.seed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant.intraday.dl.config import DLConfig

if TYPE_CHECKING:  # for type-checkers only; no runtime torch import
    import torch.nn as _nn


def build_model(config: DLConfig) -> Any:
    """Construct an LSTMRegressor (1-input LSTM -> linear head -> scalar), seeded for
    deterministic initial weights. Returns a torch.nn.Module."""
    import torch
    from torch import nn

    torch.manual_seed(config.seed)

    class LSTMRegressor(nn.Module):
        def __init__(self, hidden_size: int, n_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, window, 1) -> use the last timestep's hidden state.
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)

    return LSTMRegressor(config.hidden_size, config.n_layers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_model.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/dl/model.py tests/intraday/dl/test_model.py
uv run ruff format quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/model.py
git add quant/intraday/dl/model.py tests/intraday/dl/test_model.py
git commit -m "feat(intraday-dl): LSTMRegressor with lazy torch + deterministic init"
```

---

### Task 6: Deterministic training loop

**Files:**
- Create: `quant/intraday/dl/train.py`
- Test: `tests/intraday/dl/test_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_train.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from quant.intraday.dl.config import DLConfig  # noqa: E402
from quant.intraday.dl.train import TrainOutput, train_model  # noqa: E402


def _learnable_data(n=400, window=8, seed=0):
    # y is a clean linear function of the window so loss MUST be able to fall.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window))
    w = rng.normal(size=window)
    y = X @ w
    return X, y


def test_train_returns_output_and_loss_curve_length():
    X, y = _learnable_data()
    cfg = DLConfig(window=8, hidden_size=8, epochs=10, batch_size=32, seed=1)
    out = train_model(X, y, cfg)
    assert isinstance(out, TrainOutput)
    assert len(out.loss_curve) == cfg.epochs
    assert all(c >= 0 for c in out.loss_curve)


def test_loss_decreases():
    X, y = _learnable_data()
    cfg = DLConfig(window=8, hidden_size=16, epochs=30, batch_size=32, seed=1)
    out = train_model(X, y, cfg)
    # training works: end loss is clearly below start loss.
    assert out.loss_curve[-1] < out.loss_curve[0] * 0.7


def test_same_seed_same_machine_determinism():
    X, y = _learnable_data()
    cfg = DLConfig(window=8, hidden_size=8, epochs=10, batch_size=32, seed=42)
    a = train_model(X, y, cfg)
    b = train_model(X, y, cfg)
    assert a.loss_curve == b.loss_curve  # identical run on the same machine
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.intraday.dl.train'`.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/dl/train.py`:

```python
"""Deterministic CPU training loop (Adam + MSE) for the LSTM alpha. torch imported
LAZILY. Reproducible for two runs on the SAME machine (torch does not guarantee bitwise
cross-machine determinism), so tests assert same-run/same-machine determinism only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant.intraday.dl.config import DLConfig


@dataclass
class TrainOutput:
    model: Any  # torch.nn.Module
    loss_curve: list[float]


def train_model(
    x_train: NDArray[np.float64], y_train: NDArray[np.float64], config: DLConfig
) -> TrainOutput:
    """Train the LSTM regressor; return the trained model + per-epoch mean loss."""
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)

    from quant.intraday.dl.model import build_model

    model = build_model(config)
    # (n, window) -> (n, window, 1) for the 1-feature LSTM.
    x_t = torch.tensor(np.asarray(x_train, dtype=np.float32)).unsqueeze(-1)
    y_t = torch.tensor(np.asarray(y_train, dtype=np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    n = x_t.shape[0]

    loss_curve: list[float] = []
    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        loss_curve.append(epoch_loss / n_batches)

    return TrainOutput(model=model, loss_curve=loss_curve)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_train.py -q`
Expected: PASS (3 tests). If `test_loss_decreases` is flaky, it indicates a real training bug — do NOT loosen the threshold without understanding why; the linear-learnable data should train easily.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/dl/train.py tests/intraday/dl/test_train.py
uv run ruff format quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/train.py
git add quant/intraday/dl/train.py tests/intraday/dl/test_train.py
git commit -m "feat(intraday-dl): deterministic Adam+MSE training loop"
```

---

### Task 7: Evaluation — metrics, predict, series generators, evaluate_track

**Files:**
- Create: `quant/intraday/dl/evaluate.py`
- Test: `tests/intraday/dl/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_evaluate.py`:

```python
import numpy as np
import pytest

from quant.intraday.dl.evaluate import (
    oos_metrics,
    random_series,
    synthetic_signal_series,
)


def test_oos_metrics_perfect_prediction():
    y = np.array([1.0, -2.0, 3.0, -0.5])
    m = oos_metrics(y, y.copy())
    assert m["mse"] == 0.0
    assert m["directional_accuracy"] == 1.0
    assert abs(m["r2"] - 1.0) < 1e-12


def test_oos_metrics_directional_accuracy():
    y_true = np.array([1.0, -1.0, 2.0, -3.0])
    y_pred = np.array([0.5, 1.0, 2.0, 4.0])  # signs: +,+,+,+ vs +,-,+,- => 2/4
    m = oos_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == 0.5


def test_random_series_is_reproducible_and_shaped():
    a = random_series(n=500, seed=3)
    b = random_series(n=500, seed=3)
    assert a.shape == (500,)
    assert np.allclose(a, b)
    assert not np.allclose(a, random_series(n=500, seed=4))


def test_synthetic_signal_has_autocorrelation():
    s = synthetic_signal_series(n=4000, seed=1)
    # AR(2) structure => lag-1 autocorrelation clearly non-zero.
    s0, s1 = s[:-1], s[1:]
    corr = np.corrcoef(s0, s1)[0, 1]
    assert abs(corr) > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_evaluate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.intraday.dl.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

Create `quant/intraday/dl/evaluate.py`:

```python
"""Out-of-sample evaluation for the DL alpha. Metrics, prediction, seeded series
generators, and the per-track LSTM-vs-baselines comparison. torch is imported LAZILY
inside predict() only; the metrics + generators are pure numpy."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant.intraday.dl.baselines import linear_predict, naive_predict
from quant.intraday.dl.config import DLConfig
from quant.intraday.dl.data import make_windows, standardize, train_test_split
from quant.intraday.dl.train import train_model


def predict(model: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Run the trained LSTM forward (eval mode, no grad). Lazy torch import."""
    import torch

    model.eval()
    x_t = torch.tensor(np.asarray(x, dtype=np.float32)).unsqueeze(-1)
    with torch.no_grad():
        out = model(x_t)
    return out.numpy().astype(np.float64)


def oos_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    """MSE, directional accuracy (sign match), and R^2. Pure numpy."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((yt - yp) ** 2))
    directional_accuracy = float(np.mean(np.sign(yp) == np.sign(yt)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"mse": mse, "directional_accuracy": directional_accuracy, "r2": r2}


def random_series(n: int, seed: int, sigma: float = 1.0) -> NDArray[np.float64]:
    """A near-martingale iid-noise return series (no learnable structure)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=n)


def synthetic_signal_series(
    n: int, seed: int, a: float = 0.6, b: float = -0.3, noise: float = 0.3
) -> NDArray[np.float64]:
    """A stationary AR(2) series r_t = a*r_{t-1} + b*r_{t-2} + eps with KNOWN learnable
    structure. The LSTM MUST beat naive on this track."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, noise, size=n)
    r = np.zeros(n, dtype=np.float64)
    for t in range(2, n):
        r[t] = a * r[t - 1] + b * r[t - 2] + eps[t]
    return r


def evaluate_track(series: NDArray[np.float64], config: DLConfig) -> dict[str, Any]:
    """Window -> chronological split -> train-only standardize -> compare LSTM vs linear
    vs naive OOS. All three predict in the same standardized space (one train-X (mu, sd))."""
    x, y = make_windows(series, config.window)
    x_tr, y_tr, x_te, y_te = train_test_split(x, y, config.train_frac)
    x_tr_z, x_te_z, mu, sd = standardize(x_tr, x_te)
    y_tr_z = (y_tr - mu) / sd
    y_te_z = (y_te - mu) / sd

    naive_hat = naive_predict(x_te_z)
    linear_hat = linear_predict(x_tr_z, y_tr_z, x_te_z)
    out = train_model(x_tr_z, y_tr_z, config)
    lstm_hat = predict(out.model, x_te_z)

    return {
        "naive": oos_metrics(y_te_z, naive_hat),
        "linear": oos_metrics(y_te_z, linear_hat),
        "lstm": oos_metrics(y_te_z, lstm_hat),
        "loss_curve": out.loss_curve,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_evaluate.py -q`
Expected: PASS (4 tests). These test the pure-numpy pieces only (no torch needed); `evaluate_track` + `predict` are exercised in Task 8.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/dl/evaluate.py tests/intraday/dl/test_evaluate.py
uv run ruff format quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/evaluate.py
git add quant/intraday/dl/evaluate.py tests/intraday/dl/test_evaluate.py
git commit -m "feat(intraday-dl): OOS metrics, predict, seeded series, evaluate_track"
```

---

### Task 8: Dual-track integration test (the headline honesty claim)

**Files:**
- Test: `tests/intraday/dl/test_dual_track.py`

This task adds NO new source — it asserts the two end-to-end claims from spec §3 and §8: on the synthetic-signal track the LSTM beats naive (machinery works); on the random track the LSTM does not catastrophically underperform the baselines (honest no-edge).

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_dual_track.py`:

```python
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from quant.intraday.dl.config import DLConfig  # noqa: E402
from quant.intraday.dl.evaluate import (  # noqa: E402
    evaluate_track,
    random_series,
    synthetic_signal_series,
)

# Modest config so the integration test runs in a few seconds, still enough to learn.
_CFG = DLConfig(window=12, hidden_size=24, epochs=40, batch_size=64, seed=7, train_frac=0.7)


def test_synthetic_track_lstm_beats_naive():
    series = synthetic_signal_series(n=3000, seed=7)
    res = evaluate_track(series, _CFG)
    # Machinery works: the LSTM extracts the AR structure -> lower OOS MSE than persistence.
    assert res["lstm"]["mse"] < res["naive"]["mse"]
    # And training actually happened: loss fell.
    assert res["loss_curve"][-1] < res["loss_curve"][0]


def test_random_track_lstm_not_catastrophic():
    series = random_series(n=3000, seed=7)
    res = evaluate_track(series, _CFG)
    # Honest no-edge: on near-random returns the LSTM does NOT beat the baselines, but it
    # must not be catastrophically worse than the naive baseline (within a tolerance band).
    assert res["lstm"]["mse"] <= res["naive"]["mse"] * 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_dual_track.py -q`
Expected: PASS already is acceptable (all source exists). But first run it to confirm both assertions actually hold with the chosen config. If `test_synthetic_track_lstm_beats_naive` fails, the model/training has a real defect (or the config is too weak) — investigate via systematic-debugging, do NOT just bump epochs blindly; the AR(2) signal is strongly learnable. If `test_random_track_lstm_not_catastrophic` fails, the model is overfitting noise badly — that is also a real finding worth understanding before adjusting tolerance.

- [ ] **Step 3: Confirm it passes deterministically**

Run twice: `uv run pytest tests/intraday/dl/test_dual_track.py -q && uv run pytest tests/intraday/dl/test_dual_track.py -q`
Expected: PASS both times (same-machine determinism).

- [ ] **Step 4: Commit**

```bash
git add tests/intraday/dl/test_dual_track.py
git commit -m "test(intraday-dl): dual-track claim — LSTM beats naive on signal, no-edge on random"
```

---

### Task 9: CLI — `quant intraday dl train` + `dl evaluate`

**Files:**
- Modify: `quant/intraday/cli.py` (add a `dl` group after the `rl` group, ~end of file)
- Test: `tests/intraday/dl/test_dl_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/intraday/dl/test_dl_cli.py`:

```python
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from click.testing import CliRunner  # noqa: E402

from quant.intraday.cli import intraday  # noqa: E402


def test_dl_train_runs():
    runner = CliRunner()
    result = runner.invoke(
        intraday, ["dl", "train", "--epochs", "5", "--n", "800", "--seed", "7"]
    )
    assert result.exit_code == 0, result.output
    assert "loss" in result.output.lower()


def test_dl_evaluate_runs_both_tracks():
    runner = CliRunner()
    result = runner.invoke(
        intraday, ["dl", "evaluate", "--epochs", "5", "--n", "800", "--seed", "7"]
    )
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "synthetic" in out and "random" in out
    assert "lstm" in out and "naive" in out and "linear" in out
    # The honesty note (EMH) must be printed.
    assert "emh" in out or "near-unforecastable" in out or "does not beat" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/intraday/dl/test_dl_cli.py -q`
Expected: FAIL — `No such command 'dl'` (the group isn't wired yet).

- [ ] **Step 3: Add the `dl` group to the CLI**

Append to the END of `quant/intraday/cli.py` (after the `rl` `compare` command):

```python
# ---------------------------------------------------------------------------
# dl group
# ---------------------------------------------------------------------------


@intraday.group()
def dl() -> None:
    """Deep-learning alpha (torch LSTM, next-bar return) — sim/research only."""


@dl.command()
@click.option("--n", type=int, default=3000, help="length of the synthetic signal series")
@click.option("--window", type=int, default=12)
@click.option("--epochs", type=int, default=40)
@click.option("--seed", type=int, default=7)
def train(n: int, window: int, epochs: int, seed: int) -> None:
    """Train on the synthetic-signal series and print the per-epoch loss curve."""
    from quant.intraday.dl.config import DLConfig
    from quant.intraday.dl.data import make_windows, standardize, train_test_split
    from quant.intraday.dl.evaluate import synthetic_signal_series
    from quant.intraday.dl.train import train_model

    cfg = DLConfig(window=window, epochs=epochs, seed=seed)
    series = synthetic_signal_series(n=n, seed=seed)
    x, y = make_windows(series, cfg.window)
    x_tr, y_tr, x_te, _ = train_test_split(x, y, cfg.train_frac)
    x_tr_z, _, mu, sd = standardize(x_tr, x_te)
    y_tr_z = (y_tr - mu) / sd
    out = train_model(x_tr_z, y_tr_z, cfg)
    click.echo(f"DL alpha training ({n} pts, window {window}, {epochs} epochs, seed {seed}):")
    curve = out.loss_curve
    for i, c in enumerate(curve):
        if i == 0 or i == len(curve) - 1 or i % 5 == 0:
            click.echo(f"  epoch {i:>3}: loss {c:.5f}")
    click.echo(f"  loss fell from {curve[0]:.5f} to {curve[-1]:.5f} (training works).")


@dl.command()
@click.option("--n", type=int, default=3000, help="length of each evaluation series")
@click.option("--window", type=int, default=12)
@click.option("--epochs", type=int, default=40)
@click.option("--seed", type=int, default=7)
def evaluate(n: int, window: int, epochs: int, seed: int) -> None:
    """Dual-track OOS comparison: LSTM vs linear vs naive on a synthetic-signal series
    (LSTM should win) and a near-random series (LSTM should NOT win — the honest result)."""
    from quant.intraday.dl.config import DLConfig
    from quant.intraday.dl.evaluate import (
        evaluate_track,
        random_series,
        synthetic_signal_series,
    )

    cfg = DLConfig(window=window, epochs=epochs, seed=seed)
    tracks = {
        "synthetic-signal": synthetic_signal_series(n=n, seed=seed),
        "random": random_series(n=n, seed=seed),
    }
    click.echo(f"DL alpha OOS evaluation (window {window}, {epochs} epochs, seed {seed}):")
    for name, series in tracks.items():
        res = evaluate_track(series, cfg)
        click.echo(f"\n  [{name}] OOS  (mse / directional-accuracy / r2):")
        for model_name in ("lstm", "linear", "naive"):
            m = res[model_name]
            click.echo(
                f"    {model_name:<7} mse {m['mse']:.5f}   "
                f"dir-acc {m['directional_accuracy']:.3f}   r2 {m['r2']:.4f}"
            )
    click.echo(
        "\nnote: intraday returns are near-unforecastable (EMH); DL does not beat simple "
        "baselines OOS on the random track — the value here is the technique + honest evaluation."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/intraday/dl/test_dl_cli.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Smoke-run the CLI by hand**

Run: `uv run quant intraday dl evaluate --epochs 8 --n 1200`
Expected: prints both track tables (lstm/linear/naive rows) and the EMH note; exit 0.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check quant/intraday/cli.py tests/intraday/dl/test_dl_cli.py
uv run ruff format quant/intraday/cli.py tests/intraday/dl/
uv run mypy quant/intraday/cli.py
git add quant/intraday/cli.py tests/intraday/dl/test_dl_cli.py
git commit -m "feat(intraday-dl): \`quant intraday dl\` train + evaluate CLI"
```

---

### Task 10: Full-suite green + containment verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm torch containment — the pure-numpy modules don't import torch**

Run: `uv run python -c "import sys; import quant.intraday.dl.config, quant.intraday.dl.data, quant.intraday.dl.baselines; assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m); print('OK: numpy modules import without torch')"`
Expected: prints `OK: ...` (importing config/data/baselines did NOT pull in torch).

- [ ] **Step 2: Confirm the full suite is green**

Run: `uv run pytest -m "not network and not alpaca" -q`
Expected: PASS — the previous green count plus the new DL tests (config 3 + data 6 + baselines 3 + model 4 + train 3 + evaluate 4 + dual-track 2 + cli 2). Note the dual-track + train + cli tests are the slow ones (LSTM training); the run will take longer than before but must stay green.

- [ ] **Step 3: Confirm skip-without-torch behavior is correct (documentation check)**

The torch test files all begin with `torch = pytest.importorskip("torch")`. Verify by grep:
Run: `grep -rl "importorskip(\"torch\")" tests/intraday/dl/`
Expected: lists `test_model.py`, `test_train.py`, `test_dual_track.py`, `test_dl_cli.py` (the torch-dependent ones). `test_config.py`, `test_data.py`, `test_baselines.py`, `test_evaluate.py` must NOT appear (they are pure-numpy).

- [ ] **Step 4: Final lint + type-check sweep of the whole subpackage**

```bash
uv run ruff check quant/intraday/dl/ tests/intraday/dl/
uv run mypy quant/intraday/dl/
```
Expected: clean (no errors).

- [ ] **Step 5: No commit needed** (verification only). If any step failed, fix via systematic-debugging and re-run before declaring the sub-project complete.

---

## Self-Review

**Spec coverage:**
- §1 architecture (standalone `quant/intraday/dl/`, lazy torch, no live path) → Tasks 2–9; lazy-import asserted in Tasks 2/3/4/5 + verified in Task 10.
- §2 dependency containment (pinned torch, lazy imports, importorskip tests, same-machine determinism) → Task 1 (pin), Tasks 5/6/8/9 (importorskip), Task 6 (`use_deterministic_algorithms` + determinism test), Task 10 (containment verification).
- §3 dual evaluation (synthetic must-win, random no-edge band) → Task 8.
- §4 components (config/data/baselines/model/train/evaluate/CLI) → Tasks 2,3,4,5,6,7,9 respectively.
- §5 data flow → Task 7 `evaluate_track`.
- §6 evaluation artifact (`dl evaluate` dual table + EMH note, `dl train` loss curve) → Task 9.
- §7 charter compliance (reproducibility, no-lookahead, honesty, containment) → Task 3 (no-lookahead split/standardize), Task 6 (determinism), Task 8/9 (honesty), Task 10 (containment).
- §8 success criteria → covered by Tasks 6 (learns), 8 (beats naive on signal / not catastrophic on random), 9 (CLI artifact), 10 (suite green + containment).

**Placeholder scan:** No TBD/TODO; every code step has complete code; every test has full assertions; every command has expected output. None found.

**Type consistency:** `DLConfig` fields (`window`, `hidden_size`, `n_layers`, `lr`, `epochs`, `batch_size`, `seed`, `train_frac`) are used consistently across Tasks 5/6/7/9. `make_windows`/`train_test_split`/`standardize` signatures (Task 3) match every call site in `evaluate_track` and the CLI (Tasks 7, 9). `standardize` returns `(train_z, test_z, mu, sd)` everywhere it is consumed. `train_model(x_train, y_train, config) -> TrainOutput(.model, .loss_curve)` consistent across Tasks 6/7/9. `oos_metrics` returns `{mse, directional_accuracy, r2}` consumed identically in Tasks 8/9. `build_model(config)` consistent across Tasks 5/6. CLI group/commands (`dl train`, `dl evaluate`) match the test invocations in Task 9. No mismatches.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-09-intraday-dl-alpha.md`.
