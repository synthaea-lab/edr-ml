"""What the site trainers' floor checks share: the model they score, and the comparison of a
candidate's robustness card with its reference's within `config.FLOOR_TOLERANCE`.

`train_site_model` compares the site model with the global model on one scenario;
`train_site_rarity` compares its candidate with the rarity withheld against a reference on each
scenario of a suite. Both read a regression the same way, from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from synthaea_ml.registry.records import RobustnessCard
from synthaea_ml.training.config import FLOOR_TOLERANCE


class Scorer(Protocol):
    """What the floor checks score with: an isolation forest, or the shipped ONNX model."""

    def decision_function(self, x: ArrayLike) -> np.ndarray:
        """The anomaly score of each row of `x`; lower is more anomalous."""
        ...


@dataclass(frozen=True)
class CardComparison:
    """A candidate's robustness card on one scenario against its reference's on the same one."""

    candidate: RobustnessCard
    reference: RobustnessCard

    @property
    def escape_rate_regressed(self) -> bool:
        """Whether the candidate's escape rate is above the reference's plus the tolerance."""
        return (
            self.candidate.escape_rate
            > self.reference.escape_rate + FLOOR_TOLERANCE.escape_rate
        )

    @property
    def median_degradation_regressed(self) -> bool:
        """Whether the candidate's median score degradation is above the reference's plus the
        tolerance."""
        return (
            self.candidate.median_score_degradation
            > self.reference.median_score_degradation + FLOOR_TOLERANCE.median_degradation
        )

    @property
    def passed(self) -> bool:
        """Whether the candidate held the floor: neither metric regressed."""
        return not (self.escape_rate_regressed or self.median_degradation_regressed)
