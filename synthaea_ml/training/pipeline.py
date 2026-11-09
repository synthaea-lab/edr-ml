"""The steps every Isolation Forest trainer shares: split, fit, calibrate, export, evaluate.

A trainer keeps its own loader, its configuration and its command line, and calls these
steps in order:

1. `split_for_calibration` holds out the conformal calibration set (issue #46).
2. `fit_calibrated_forest` fits the forest, sets the conformal threshold and computes the
   feature bounds of the training rows.
3. `export_onnx` and `write_scorer_metadata` write what the Rust scorer loads.
4. `evaluate_robustness` runs the adversarial scenario suite (issue #45).

Writing the training record stays in each trainer, because what it binds (the dataset
versions, the extra fields) is trainer-specific.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from synthaea_ml.calibration import (
    ConformalCalibration,
    FeatureBounds,
    calibrate_threshold,
    compute_feature_bounds,
)
from synthaea_ml.evaluation.robustness import run_robustness_evaluation
from synthaea_ml.registry.records import RobustnessCard
from synthaea_ml.training.config import (
    CALIBRATION,
    RANDOM_SEED,
    CalibrationConfig,
    IsolationForestConfig,
)

CALIBRATION_FRACTION = 0.3
"""Share of the rows held out of training to set the conformal threshold (a 70/30 split)."""

ONNX_TARGET_OPSET = {"": 18, "ai.onnx.ml": 3}
"""skl2onnx 1.20 does not yet follow the `ai.onnx.ml` v4 domain that onnx 1.22 emits by
default, so the export is pinned to the latest version this skl2onnx can consume."""


@dataclass(frozen=True)
class CalibratedForest:
    """A fitted forest with the conformal threshold and the feature bounds the scorer needs."""

    model: IsolationForest
    calibration: ConformalCalibration
    bounds: FeatureBounds


def split_for_calibration(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The seeded `(training, calibration)` split of the rows of `x`."""
    x_train, x_cal = train_test_split(x, test_size=CALIBRATION_FRACTION, random_state=RANDOM_SEED)
    return x_train, x_cal


def fit_calibrated_forest(
    x_train: np.ndarray,
    x_cal: np.ndarray,
    feature_names: Sequence[str],
    forest: IsolationForestConfig,
    calibration: CalibrationConfig = CALIBRATION,
) -> CalibratedForest:
    """Fit `forest` on `x_train`, calibrate its threshold on `x_cal` against the FP budget,
    and bound each feature by the range seen in `x_train` (plus the configured margin)."""
    model = IsolationForest(**forest.hyperparameters())
    model.fit(x_train)
    threshold = calibrate_threshold(
        model,
        x_cal,
        fp_budget=calibration.fp_budget_per_endpoint_day,
        benign_rate_per_day=calibration.benign_rate_per_day,
    )
    bounds = compute_feature_bounds(
        x_train, list(feature_names), margin=calibration.feature_bounds_margin
    )
    return CalibratedForest(model=model, calibration=threshold, bounds=bounds)


def export_onnx(model: IsolationForest, sample: np.ndarray, path: Path) -> None:
    """Write `model` to `path` as ONNX; `sample` (one or more rows) fixes the input shape."""
    onnx_model = to_onnx(model, sample[:1], target_opset=ONNX_TARGET_OPSET)
    path.write_bytes(onnx_model.SerializeToString())


def write_scorer_metadata(
    path: Path, calibration: ConformalCalibration, bounds: FeatureBounds
) -> None:
    """Write the threshold and feature bounds the Rust scorer reads next to the model."""
    metadata = {
        "threshold": calibration.threshold,
        "feature_bounds": {
            "feature_names": bounds.feature_names,
            "min_values": bounds.min_values,
            "max_values": bounds.max_values,
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def evaluate_robustness(
    model: IsolationForest,
    scenarios: Sequence[Path],
    events: Path | None,
    tier: str,
) -> list[RobustnessCard]:
    """One robustness card per scenario, in order.

    A failed evaluation propagates, with a note naming the scenario: a crashed evaluation
    must not produce a record with missing cards and exit 0.
    """
    cards: list[RobustnessCard] = []
    for scenario in scenarios:
        try:
            card = run_robustness_evaluation(
                model=model,
                scenario_yaml=scenario,
                tier=tier,
                mutation_seed=RANDOM_SEED,
                events_source=events,
            )
        except Exception as error:
            error.add_note(f"robustness evaluation failed for {scenario}")
            raise
        cards.append(card)
    return cards


def describe_card(card: RobustnessCard) -> str:
    """One line summarizing a robustness card, as the trainers print it."""
    return (
        f"  {card.scenario_name}: escape_rate={card.escape_rate:.2%}, "
        f"median_degradation={card.median_score_degradation:+.3f}"
    )
