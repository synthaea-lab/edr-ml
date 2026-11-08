"""The shared training configuration keeps the values the trainers were tuned and recorded with.

A shipped model is reproducible from its record only if these numbers do not move silently: a
change here is a retraining decision, made on purpose and with this test updated alongside.
"""

from __future__ import annotations

from synthaea_ml.training.config import (
    BEHAVIOR_FOREST,
    CALIBRATION,
    CMDLINE_FOREST,
    CORRELATION_FOREST,
    FLOOR_TOLERANCE,
    RANDOM_SEED,
)


def test_each_forest_family_keeps_its_recorded_hyperparameters() -> None:
    assert CMDLINE_FOREST.hyperparameters() == {
        "n_estimators": 100,
        "contamination": 0.05,
        "random_state": 42,
    }
    assert BEHAVIOR_FOREST.hyperparameters() == {
        "n_estimators": 100,
        "contamination": 0.05,
        "random_state": 42,
    }
    assert CORRELATION_FOREST.hyperparameters() == {
        "n_estimators": 100,
        "contamination": 0.02,
        "random_state": 42,
    }


def test_hyperparameters_keep_the_key_order_of_the_model_record() -> None:
    assert list(CMDLINE_FOREST.hyperparameters()) == ["n_estimators", "contamination", "random_state"]


def test_the_calibration_contract_keeps_its_fp_budget_rate_and_margin() -> None:
    assert CALIBRATION.fp_budget_per_endpoint_day == 5.0
    assert CALIBRATION.benign_rate_per_day == 1000.0
    assert CALIBRATION.feature_bounds_margin == 0.05


def test_the_floor_tolerances_and_seed_keep_their_values() -> None:
    assert FLOOR_TOLERANCE.escape_rate == 0.05
    assert FLOOR_TOLERANCE.median_degradation == 0.10
    assert RANDOM_SEED == 42
