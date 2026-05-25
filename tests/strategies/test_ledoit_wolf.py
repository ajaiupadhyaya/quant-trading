"""Tests for the Ledoit-Wolf shrinkage estimator used by HRP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.risk_parity import ledoit_wolf_shrinkage


def _gaussian_panel(n: int, k: int, corr: float, seed: int) -> pd.DataFrame:
    """Sample n daily returns for k assets with constant pairwise correlation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, size=(n, 1))
    idio = rng.normal(0, 0.01, size=(n, k))
    rho = np.sqrt(corr)
    panel = rho * base + np.sqrt(1 - corr) * idio
    return pd.DataFrame(panel, columns=[f"A{i}" for i in range(k)])


def test_lw_returns_psd_matrix() -> None:
    panel = _gaussian_panel(n=120, k=8, corr=0.3, seed=0)
    shrunk, delta = ledoit_wolf_shrinkage(panel)
    eigs = np.linalg.eigvalsh(shrunk.values)
    assert (eigs > -1e-10).all()
    assert 0.0 <= delta <= 1.0


def test_lw_preserves_diagonal_variances() -> None:
    """LW shrinks to a constant-correlation target; diagonals (variances) are preserved."""
    panel = _gaussian_panel(n=120, k=6, corr=0.2, seed=1)
    shrunk, _ = ledoit_wolf_shrinkage(panel)
    sample = panel.cov()
    np.testing.assert_allclose(
        np.diag(shrunk.values),
        np.diag(sample.values),
        rtol=1e-9,
        atol=1e-12,
    )


def test_lw_high_correlation_results_in_high_shrinkage() -> None:
    """When sample correlations are noisy (small n vs k), δ should be meaningfully > 0."""
    panel = _gaussian_panel(n=40, k=10, corr=0.1, seed=2)
    _, delta = ledoit_wolf_shrinkage(panel)
    assert delta > 0.05


def test_lw_handles_short_or_degenerate_input() -> None:
    short = _gaussian_panel(n=3, k=4, corr=0.0, seed=3)
    shrunk, delta = ledoit_wolf_shrinkage(short)
    # Falls back to sample cov; delta = 0.
    assert delta == 0.0
    assert shrunk.shape == (4, 4)
