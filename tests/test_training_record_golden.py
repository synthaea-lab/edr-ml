"""Golden bytes of ``model_record.json``: the release gate hashes these files, so the
writer's output and the reader's accepted inputs are an on-disk contract.

One schema v3 record carries every optional block (scenario replays, robustness cards,
conformal calibration, feature bounds) and is pinned byte for byte. A schema v2 record,
written before the v3 blocks existed, must still load.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from synthaea_ml.registry.training_record import (
    MODEL_RECORD_FILENAME,
    ConformalCalibration,
    DatasetVersion,
    Environment,
    ExpectedDetection,
    FeatureBounds,
    MutationTestResult,
    ObservedDetection,
    RobustnessCard,
    ScenarioReplayResult,
    TrainingRecord,
    load_training_record,
    write_training_record,
)

_TRAINED_AT = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)
_BASELINE_SHA = "ab" * 32
_SCENARIO_SHA = "cd" * 32

_DATASET = DatasetVersion(
    name="linux__dev__abc__2026-09-11", baseline_sha256=_BASELINE_SHA, sample_count=2
)
_REPLAY = ScenarioReplayResult(
    scenario_name="reverse_shell",
    scenario_yaml_sha256=_SCENARIO_SHA,
    run_at="2026-09-17T10:15:00Z",
    environment=Environment(os="linux", kernel="6.6.0-generic", arch="x86_64"),
    expected_detections=[
        ExpectedDetection(technique="T1059", rule="cmdline-anomaly", min_count=1, tolerance=2)
    ],
    observed_detections=[ObservedDetection(technique="T1059", rule="cmdline-anomaly", count=2)],
    passed=True,
)
_CARD = RobustnessCard(
    scenario_name="reverse_shell",
    scenario_yaml_sha256=_SCENARIO_SHA,
    tested_at="2026-09-23T10:15:00Z",
    mutation_results=[
        MutationTestResult(
            mutation_class="ArgvPadding",
            intensity="light",
            original_score=-0.25,
            mutated_score=-0.125,
            score_delta=0.125,
            threshold=-0.0421,
            escaped=False,
            seed=42,
        )
    ],
    escape_rate=0.0,
    median_score_degradation=0.125,
    worst_case_degradation=0.125,
)
_CALIBRATION = ConformalCalibration(
    fp_budget_per_endpoint_day=5.0,
    threshold=-0.0421,
    calibration_set_size=300,
    benign_baseline_rate=1000.0,
    calibrated_at="2026-09-11T11:59:00+00:00",
)
_BOUNDS = FeatureBounds(
    feature_names=["argv_len", "entropy"], min_values=[0.0, -0.1], max_values=[12.6, 5.25]
)

GOLDEN_V3_RECORD = """{
  "conformal_calibration": {
    "benign_baseline_rate": 1000.0,
    "calibrated_at": "2026-09-11T11:59:00+00:00",
    "calibration_set_size": 300,
    "fp_budget_per_endpoint_day": 5.0,
    "threshold": -0.0421
  },
  "dataset_versions": [
    {
      "baseline_sha256": "abababababababababababababababababababababababababababababababab",
      "name": "linux__dev__abc__2026-09-11",
      "sample_count": 2
    }
  ],
  "extra": {
    "commit_sha": "deadbeef"
  },
  "feature_bounds": {
    "feature_names": [
      "argv_len",
      "entropy"
    ],
    "max_values": [
      12.6,
      5.25
    ],
    "min_values": [
      0.0,
      -0.1
    ]
  },
  "hyperparameters": {
    "contamination": 0.05,
    "n_estimators": 100
  },
  "robustness_cards": [
    {
      "escape_rate": 0.0,
      "median_score_degradation": 0.125,
      "mutation_results": [
        {
          "escaped": false,
          "intensity": "light",
          "mutated_score": -0.125,
          "mutation_class": "ArgvPadding",
          "original_score": -0.25,
          "score_delta": 0.125,
          "seed": 42,
          "threshold": -0.0421
        }
      ],
      "scenario_name": "reverse_shell",
      "scenario_yaml_sha256": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
      "tested_at": "2026-09-23T10:15:00Z",
      "worst_case_degradation": 0.125
    }
  ],
  "scenario_replays": [
    {
      "environment": {
        "arch": "x86_64",
        "kernel": "6.6.0-generic",
        "os": "linux"
      },
      "expected_detections": [
        {
          "min_count": 1,
          "rule": "cmdline-anomaly",
          "technique": "T1059",
          "tolerance": 2
        }
      ],
      "observed_detections": [
        {
          "count": 2,
          "rule": "cmdline-anomaly",
          "technique": "T1059"
        }
      ],
      "passed": true,
      "run_at": "2026-09-17T10:15:00Z",
      "scenario_name": "reverse_shell",
      "scenario_yaml_sha256": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"
    }
  ],
  "schema_version": 3,
  "trained_at": "2026-09-11T12:00:00Z",
  "training_script": "synthaea_ml/training/train_linux.py"
}
"""

_GOLDEN_V2 = """{
  "dataset_versions": [
    {
      "baseline_sha256": "abababababababababababababababababababababababababababababababab",
      "name": "linux__dev__abc__2026-09-11",
      "sample_count": 2
    }
  ],
  "extra": {},
  "hyperparameters": {
    "n_estimators": 100
  },
  "scenario_replays": [],
  "schema_version": 2,
  "trained_at": "2026-09-11T12:00:00Z",
  "training_script": "synthaea_ml/training/train_linux.py"
}
"""


def _write_full_v3_record(model_dir: Path) -> TrainingRecord:
    return write_training_record(
        model_dir,
        training_script="synthaea_ml/training/train_linux.py",
        dataset_versions=[_DATASET],
        scenario_replays=[_REPLAY],
        robustness_cards=[_CARD],
        hyperparameters={"n_estimators": 100, "contamination": 0.05},
        trained_at=_TRAINED_AT,
        extra={"commit_sha": "deadbeef"},
        conformal_calibration=_CALIBRATION,
        feature_bounds=_BOUNDS,
    )


def test_v3_record_with_every_optional_block_is_byte_identical_to_golden(
    tmp_path: Path,
) -> None:
    """Text mode writes CRLF on Windows; normalising it keeps the check portable while
    still failing on any change to content or key order."""
    _write_full_v3_record(tmp_path)

    written = (tmp_path / MODEL_RECORD_FILENAME).read_bytes().replace(b"\r\n", b"\n")
    assert written == GOLDEN_V3_RECORD.encode("utf-8")


def test_v3_golden_loads_into_the_record_that_wrote_it(tmp_path: Path) -> None:
    golden_dir, written_dir = tmp_path / "golden", tmp_path / "written"
    golden_dir.mkdir()
    written_dir.mkdir()
    (golden_dir / MODEL_RECORD_FILENAME).write_text(GOLDEN_V3_RECORD, encoding="utf-8")

    assert load_training_record(golden_dir) == _write_full_v3_record(written_dir)


def test_v2_golden_still_loads_with_empty_v3_blocks(tmp_path: Path) -> None:
    (tmp_path / MODEL_RECORD_FILENAME).write_text(_GOLDEN_V2, encoding="utf-8")

    assert load_training_record(tmp_path) == TrainingRecord(
        schema_version=2,
        trained_at="2026-09-11T12:00:00Z",
        training_script="synthaea_ml/training/train_linux.py",
        dataset_versions=[_DATASET],
        hyperparameters={"n_estimators": 100},
    )
