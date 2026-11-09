"""The exact failure messages of the two rarity floors (issue #640).

A run that fails a floor prints these to the operator, and both floors share one
implementation, so this pins that each still names its own reference and consequence.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from synthaea_ml.training import train_site_rarity as rare


class _Constant:
    """A model whose score is the same for every row."""

    def __init__(self, score: float, threshold: float = 0.0) -> None:
        self.score = score
        self.threshold = threshold
        self.threshold_source = "model_metadata.json"

    def decision_function(self, x: object) -> np.ndarray:
        return np.full(len(np.asarray(x)), self.score)


X_TEST = np.zeros((4, 9), dtype=np.float32)
FLAGS_ALL = _Constant(-1.0)
FLAGS_NONE = _Constant(1.0)
AT_ZERO = SimpleNamespace(threshold=0.0)


def test_the_cmdline_only_floor_names_its_reference_and_what_failing_means() -> None:
    floor = rare.check_rarity_floor(FLAGS_ALL, AT_ZERO, FLAGS_NONE, AT_ZERO, X_TEST)

    assert not floor.passed
    assert (floor.fp_reference, floor.fp_withheld) == (0.0, 1.0)
    assert floor.failures == [
        "with the rarity withheld the candidate flags 100.00% of the benign test slice, above "
        "the cmdline-only model's 0.00% + 1.00%: a fleet with no snapshot yet would see that "
        "false-positive rate"
    ]


def test_the_global_floor_names_its_reference_and_what_failing_means() -> None:
    floor = rare.check_global_floor(FLAGS_ALL, AT_ZERO, FLAGS_NONE, X_TEST)

    assert not floor.passed
    assert (floor.fp_reference, floor.fp_withheld) == (0.0, 1.0)
    assert floor.failures == [
        "with the rarity withheld the candidate flags 100.00% of the benign test slice, above "
        "the global model's 0.00% + 1.00%: a fleet with no snapshot yet would see more false "
        "positives than with the global model"
    ]


def test_an_empty_test_slice_flags_nothing_and_passes() -> None:
    empty = np.zeros((0, 9), dtype=np.float32)

    floor = rare.check_rarity_floor(FLAGS_ALL, AT_ZERO, FLAGS_ALL, AT_ZERO, empty)

    assert floor.passed
    assert (floor.fp_reference, floor.fp_withheld) == (0.0, 0.0)
