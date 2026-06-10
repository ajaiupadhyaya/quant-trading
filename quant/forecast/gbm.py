"""Deterministic gradient-boosted regression trees (pure numpy, no dependencies).

Hand-rolled to match the codebase's model ethos (HMM/Kalman/HAR/ridge are all
from-scratch numpy): fully deterministic given ``(x, y, config)``, transparent,
PIT-stable, and dependency-free. Used only by the DSR-gated, research-only
cross-sectional alpha evaluation in :mod:`quant.forecast.factor` — it is wired to
no strategy, tilt, or order path.

Squared-error regression trees (depth-limited, ``min_samples_leaf``-bounded,
exhaustive midpoint split search) boosted with shrinkage and optional seeded row
subsampling. Small panels, so the naive split search is plenty fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GBMConfig:
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 5
    subsample: float = 1.0  # per-round row subsample fraction (seeded)
    seed: int = 0


@dataclass(frozen=True)
class _Node:
    # Leaf nodes carry ``value``; internal nodes carry ``feature``/``threshold``
    # and child indices into the owning tree's node list.
    feature: int
    threshold: float
    left: int
    right: int
    value: float
    is_leaf: bool


@dataclass(frozen=True)
class _Tree:
    nodes: tuple[_Node, ...]


@dataclass(frozen=True)
class GBMModel:
    base: float
    learning_rate: float
    trees: tuple[_Tree, ...] = field(default_factory=tuple)


def _best_split(x: np.ndarray, y: np.ndarray, min_samples_leaf: int) -> tuple[int, float] | None:
    """Feature + threshold minimizing child SSE, or ``None`` if no valid split.

    The per-feature split-gain over every candidate position is computed
    vectorially (cumulative sums + a single numpy gain array), so there is no
    Python loop over rows — essential for refitting on the cumulative panel each
    walk-forward month.
    """
    n, p = x.shape
    if n < 2 * min_samples_leaf:
        return None
    parent_sse = float(((y - y.mean()) ** 2).sum())
    # Candidate left-sizes i+1 for split positions i in [msl-1, n-msl-1].
    lo = min_samples_leaf - 1
    hi = n - min_samples_leaf  # exclusive upper bound on i
    if hi <= lo:
        return None
    pos = np.arange(lo, hi)
    n_l = (pos + 1).astype(float)
    n_r = (n - (pos + 1)).astype(float)
    best_gain = 1e-12
    best: tuple[int, float] | None = None
    for j in range(p):
        col = x[:, j]
        order = np.argsort(col, kind="mergesort")
        col_s = col[order]
        y_s = y[order]
        csum = np.cumsum(y_s)
        csqsum = np.cumsum(y_s * y_s)
        total = csum[-1]
        total_sq = csqsum[-1]
        sum_l = csum[pos]
        sse_l = csqsum[pos] - sum_l * sum_l / n_l
        sum_r = total - sum_l
        sse_r = (total_sq - csqsum[pos]) - sum_r * sum_r / n_r
        gain = parent_sse - (sse_l + sse_r)
        # Disallow splitting between equal feature values.
        gain = np.where(col_s[pos] == col_s[pos + 1], -np.inf, gain)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain = float(gain[k])
            i = int(pos[k])
            best = (j, float((col_s[i] + col_s[i + 1]) / 2.0))
    return best


def _build_tree(x: np.ndarray, y: np.ndarray, cfg: GBMConfig) -> _Tree:
    nodes: list[_Node] = []

    def grow(idx: np.ndarray, depth: int) -> int:
        yv = y[idx]
        value = float(yv.mean()) if idx.size else 0.0
        if depth >= cfg.max_depth or idx.size < 2 * cfg.min_samples_leaf:
            nodes.append(_Node(-1, 0.0, -1, -1, value, True))
            return len(nodes) - 1
        split = _best_split(x[idx], yv, cfg.min_samples_leaf)
        if split is None:
            nodes.append(_Node(-1, 0.0, -1, -1, value, True))
            return len(nodes) - 1
        feat, thr = split
        mask = x[idx, feat] <= thr
        left_idx, right_idx = idx[mask], idx[~mask]
        # Reserve this internal node's slot before growing children.
        slot = len(nodes)
        nodes.append(_Node(feat, thr, -1, -1, value, False))
        left = grow(left_idx, depth + 1)
        right = grow(right_idx, depth + 1)
        nodes[slot] = _Node(feat, thr, left, right, value, False)
        return slot

    grow(np.arange(x.shape[0]), 0)
    return _Tree(nodes=tuple(nodes))


def _predict_tree(tree: _Tree, x: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=float)
    nodes = tree.nodes
    for r in range(x.shape[0]):
        node = nodes[0]
        while not node.is_leaf:
            node = nodes[node.left] if x[r, node.feature] <= node.threshold else nodes[node.right]
        out[r] = node.value
    return out


def fit_gbm(x: np.ndarray, y: np.ndarray, config: GBMConfig | None = None) -> GBMModel:
    """Fit gradient-boosted regression trees. Deterministic for fixed inputs."""
    cfg = config or GBMConfig()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    base = float(y.mean()) if n else 0.0
    pred = np.full(n, base)
    trees: list[_Tree] = []
    sub = min(max(cfg.subsample, 0.0), 1.0)
    for k in range(cfg.n_estimators):
        residual = y - pred
        if sub < 1.0 and n > 0:
            rng = np.random.default_rng(cfg.seed + k)
            m = max(2 * cfg.min_samples_leaf, round(sub * n))
            rows = rng.choice(n, size=min(m, n), replace=False)
            tree = _build_tree(x[rows], residual[rows], cfg)
        else:
            tree = _build_tree(x, residual, cfg)
        trees.append(tree)
        pred = pred + cfg.learning_rate * _predict_tree(tree, x)
    return GBMModel(base=base, learning_rate=cfg.learning_rate, trees=tuple(trees))


def predict_gbm(model: GBMModel, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape[0], model.base)
    for tree in model.trees:
        out = out + model.learning_rate * _predict_tree(tree, x)
    return out
