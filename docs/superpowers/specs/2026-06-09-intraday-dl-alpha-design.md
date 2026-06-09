# Intraday DL Alpha (LSTM, next-bar return) — Design Spec

**Date:** 2026-06-09
**Status:** Approved design, pending implementation plan
**Scope:** Sub-project D of the intraday/60s showcase track. Independent of the spine (0), A, B, C; sim/research only.

---

## Context

The intraday showcase track has shipped 0 (live sleeve loop), A (Almgren–Chriss execution), B
(Avellaneda–Stoikov market making), and C (tabular Q-learning execution). This sub-project adds a
**deep-learning alpha model**: a torch LSTM predicting next-bar return.

**Goal = portfolio/learning showcase** (see [[project-quant-trading-intraday]]). Decisions locked
in brainstorming:
1. **torch LSTM** (real recurrent net), NOT a numpy hand-rolled net — the owner chose the literal
   architecture for the showcase, accepting the dependency. (numpy-only MLP and Transformer were
   the alternatives.)
2. **Task = next-bar return regression**, evaluated OUT-OF-SAMPLE against a linear (OLS) and a
   naive baseline. Sim/research only, no live wiring.

**Operational note:** the current machine is the MacBook dev clone, NOT the M4. The owner will pull
this onto the M4 and `uv sync` to make it "ready to run." So torch is a PINNED dependency installed
by `uv sync` on the M4 (Apple Silicon CPU wheel), and the module must be contained so the rest of
the repo never breaks on it (see §2).

**Honest framing (recorded, surfaced in CLI):** intraday returns are near-unforecastable (cf. the
documented ARIMA EMH-negative result in the charter). The rigorous question is "does the LSTM beat
simple baselines out-of-sample?" — expected NO on real-style returns. The showcase value is the DL
machinery + a rigorous honest evaluation, not a discovered edge.

---

## 1. Architecture

A new standalone `quant/intraday/dl/` subpackage: data windowing, a torch `LSTMRegressor`, a
deterministic training loop, dependency-free baselines (naive + numpy OLS), an out-of-sample
evaluator, and a CLI. It imports nothing from the live loop and touches no live path. torch is
imported lazily (inside functions) so importing `quant.*` elsewhere never pays the torch cost.

---

## 2. Dependency handling (torch containment — the M4-readiness requirement)

- `torch` is added to `pyproject.toml` dependencies, pinned. `git pull && uv sync` on the M4
  installs the Apple-Silicon CPU wheel. (We also `uv sync` on this MacBook so the tests run here.)
- **Lazy imports:** every `import torch` lives INSIDE the function/method that needs it (in
  `model.py`/`train.py`), never at module top of anything imported on the common path. Importing
  `quant.intraday.dl.config`/`data`/`baselines` must NOT import torch.
- **Skippable tests:** test modules that need torch begin with `torch = pytest.importorskip("torch")`,
  so on any machine without torch they SKIP (not fail). The existing full suite stays green
  regardless of torch's presence.
- **Determinism scope:** CPU-only, `torch.manual_seed(seed)` + `torch.use_deterministic_algorithms(True)`.
  This gives reproducible results for two runs ON THE SAME MACHINE; torch does NOT guarantee bitwise
  cross-machine determinism, so tests assert SAME-RUN/SAME-MACHINE determinism only.

---

## 3. The dual evaluation (machinery works AND honesty about edge)

Two tracks, both run by `evaluate.py`:

- **Sanity track (synthetic-with-signal):** a synthetic series with a KNOWN learnable autoregressive
  structure (e.g. `r_t = a·r_{t-1} + b·r_{t-2} + noise`). The LSTM MUST beat the naive baseline OOS
  here — proving training/backprop work. A test asserts this.
- **Honest track (near-random returns):** a (near-)martingale return series (iid noise). The LSTM is
  expected to NOT beat the baselines OOS. The test asserts the LSTM is within a tolerance band of the
  baselines (not catastrophically worse) — asserting honesty, NOT a fabricated edge. The CLI prints
  the EMH note.

---

## 4. Components

`quant/intraday/dl/` (new subpackage):

| File | Responsibility |
|---|---|
| `config.py` | `DLConfig`: `window`, `hidden_size`, `n_layers`, `lr`, `epochs`, `batch_size`, `seed`, `train_frac`. Validated; no magic numbers. NO torch import. |
| `data.py` | `make_windows(series, window) -> (X, y)`: sliding lagged windows where `X[i]` is `series[i:i+window]` and `y[i]` is `series[i+window]` (next value). `train_test_split(X, y, train_frac)`: chronological (no shuffle). `standardize(train, test)`: z-score using TRAIN mean/std only (no lookahead). Returns numpy arrays. NO torch import. |
| `baselines.py` | `naive_predict(X) -> y_hat`: **persistence** — predict the last in-window value `X[:, -1]` (a meaningful baseline on both tracks: on the AR-signal series it captures autocorrelation, on returns it is the random-walk-on-returns guess). `linear_predict(X_train, y_train, X_test) -> y_hat`: numpy OLS via `np.linalg.lstsq`. Dependency-free. NO torch. |
| `model.py` | `LSTMRegressor` (torch.nn.Module: 1-layer LSTM `hidden_size` + linear head → scalar). Lazy `import torch` inside; built deterministically from `config.seed`. |
| `train.py` | `train_model(X_train, y_train, config) -> TrainOutput` (the trained model + per-epoch loss curve). Adam + MSE, CPU, seeded + deterministic. Lazy torch. |
| `evaluate.py` | `predict(model, X)`; `oos_metrics(y_true, y_pred) -> {mse, directional_accuracy, r2}`; `evaluate_track(series, config) -> dict` comparing LSTM vs linear vs naive on one series; `synthetic_signal_series(...)` and `random_series(...)` generators (seeded). |
| (CLI) | `quant intraday dl train` + `quant intraday dl evaluate`, added to `quant/intraday/cli.py` (lazy imports inside the command bodies). |

---

## 5. Data flow

`series (1-D returns) → make_windows → chronological split → standardize(train-only) → train_model
(LSTM) → predict on test → oos_metrics`. Baselines consume the same windows/splits. The evaluator
runs this for both the synthetic-signal series and the random series and tabulates LSTM vs linear vs
naive.

---

## 6. Evaluation artifact (the headline)

- `quant intraday dl evaluate [--seed S]`: prints, for BOTH tracks, an OOS table of MSE /
  directional-accuracy / R² for **LSTM vs linear vs naive**. Expected: LSTM clearly wins on the
  synthetic-signal track (machinery works); LSTM ≈ baselines on the random track (no edge). Prints
  the note: "intraday returns are near-unforecastable (EMH); DL does not beat simple baselines OOS —
  the value here is the technique + honest evaluation."
- `quant intraday dl train [--seed S]`: trains on the synthetic-signal series and prints the
  per-epoch loss curve (showing it decreases — training works).

Both default to seeded synthetic series so they run without live data.

---

## 7. Charter compliance

- **Reproducibility:** seeded numpy generators + `torch.manual_seed` + deterministic algorithms;
  same-machine runs reproduce. Config-driven; no magic numbers.
- **No lookahead:** chronological split; standardization uses train statistics only; windows use only
  past values to predict the next.
- **Honesty:** spec + CLI state plainly that the LSTM does not beat simple baselines on intraday
  returns OOS (EMH, cf. ARIMA). The synthetic-signal track exists precisely to separate "the
  machinery works" from "there is alpha" — and only the former is claimed.
- **Containment:** torch is a deliberate, pinned dep with lazy imports and skippable tests so the
  rest of the system is unaffected.

---

## 8. Success criteria

- torch is pinned in `pyproject.toml`; `uv sync` installs it; importing `quant.intraday.dl.config`/
  `data`/`baselines` does NOT import torch (lazy); DL tests skip cleanly where torch is absent.
- `make_windows`/split/standardize are correct and lookahead-free (train-only stats).
- Training is deterministic on a given machine (same seed ⇒ identical predictions) and **learns**
  (loss decreases; LSTM beats naive OOS on the synthetic-signal track).
- On the random track, the LSTM does NOT catastrophically underperform the baselines (within a stated
  tolerance) — the honest no-edge result.
- `quant intraday dl evaluate` produces the dual-track LSTM-vs-baselines table with the EMH note.
- The full suite (excluding network/alpaca, and skipping torch tests where torch is absent) stays
  green; the only new runtime dependency is torch.

---

## 9. Out of scope / deferred

Transformer architecture, GPU/MPS acceleration + tuning, hyperparameter search, walk-forward CV,
multi-asset/cross-sectional features, live wiring, and any real-capital use. Each a possible later
sub-step with its own spec.
