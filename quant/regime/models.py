"""Frozen value types and label constants for the regime engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIME_LABELS: tuple[str, str, str] = ("calm-bull", "choppy", "crisis")
N_STATES: int = 3


@dataclass(frozen=True, eq=False)
class HMMParams:
    """Parameters of a diagonal-covariance Gaussian HMM.

    Shapes: start_prob (K,), trans_mat (K, K), means (K, F), variances (K, F).
    K = number of hidden states, F = number of features.
    """

    start_prob: np.ndarray
    trans_mat: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def to_json_dict(self) -> dict[str, object]:
        return {
            "start_prob": self.start_prob.tolist(),
            "trans_mat": self.trans_mat.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> HMMParams:
        return cls(
            start_prob=np.asarray(payload["start_prob"], dtype=float),
            trans_mat=np.asarray(payload["trans_mat"], dtype=float),
            means=np.asarray(payload["means"], dtype=float),
            variances=np.asarray(payload["variances"], dtype=float),
        )
