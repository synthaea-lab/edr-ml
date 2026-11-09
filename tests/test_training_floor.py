"""A floor's robustness comparison: a candidate regresses only past the reference plus the
shared tolerance, and each metric is judged on its own."""

from __future__ import annotations

import pytest
from helpers import robustness_card

from synthaea_ml.training.config import FLOOR_TOLERANCE
from synthaea_ml.training.floor import CardComparison

_REFERENCE = robustness_card(0.10, 0.20)


def _against_reference(escape: float, degradation: float) -> CardComparison:
    return CardComparison(candidate=robustness_card(escape, degradation), reference=_REFERENCE)


def test_a_candidate_exactly_at_the_tolerance_holds_the_floor() -> None:
    comparison = _against_reference(
        0.10 + FLOOR_TOLERANCE.escape_rate, 0.20 + FLOOR_TOLERANCE.median_degradation
    )

    assert not comparison.escape_rate_regressed
    assert not comparison.median_degradation_regressed
    assert comparison.passed


@pytest.mark.parametrize(
    ("escape", "degradation", "escape_regressed", "degradation_regressed"),
    [
        (0.16, 0.20, True, False),
        (0.10, 0.31, False, True),
        (0.16, 0.31, True, True),
    ],
)
def test_a_candidate_past_the_tolerance_fails_on_the_metric_that_regressed(
    escape: float, degradation: float, escape_regressed: bool, degradation_regressed: bool
) -> None:
    comparison = _against_reference(escape, degradation)

    assert comparison.escape_rate_regressed is escape_regressed
    assert comparison.median_degradation_regressed is degradation_regressed
    assert not comparison.passed
