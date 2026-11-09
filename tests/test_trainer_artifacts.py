"""Pins the bytes each trainer writes: `model.onnx`, `model_metadata.json` and the record.

The reference below rebuilds the artifacts from the raw scikit-learn, skl2onnx and
calibration calls, the way the trainers did before they shared `training/pipeline.py`. The
trainers must produce the same bytes, so a refactor of the shared steps cannot change a
shipped model. The reference is computed in the same run rather than stored as a hash, so
the pin holds across scikit-learn versions and operating systems.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from helpers import write_baseline_dataset
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from test_train_behavior import _capture
from test_train_behavior import _write_dataset as _write_behavior_dataset

from synthaea_ml.calibration import calibrate_threshold, compute_feature_bounds
from synthaea_ml.data.baselines import load_cmdlines, unique_in_order
from synthaea_ml.features import cmdline, t1
from synthaea_ml.training import train_behavior, train_linux, train_windows

_START = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)
_END = datetime(2026, 9, 11, 11, 0, 0, tzinfo=UTC)
_FOREST = {"n_estimators": 100, "contamination": 0.05, "random_state": 42}


def _reference(x: np.ndarray, feature_names: list[str]) -> tuple[bytes, str, dict]:
    """The ONNX bytes, the metadata text and the calibration/bounds of the record."""
    x_train, x_cal = train_test_split(x, test_size=0.3, random_state=42)
    clf = IsolationForest(**_FOREST).fit(x_train)
    cal = calibrate_threshold(clf, x_cal, fp_budget=5.0, benign_rate_per_day=1000.0)
    bounds = compute_feature_bounds(x_train, feature_names, margin=0.05)
    onnx_bytes = to_onnx(
        clf, x_train[:1], target_opset={"": 18, "ai.onnx.ml": 3}
    ).SerializeToString()
    bounds_dict = {
        "feature_names": bounds.feature_names,
        "min_values": bounds.min_values,
        "max_values": bounds.max_values,
    }
    metadata = json.dumps({"threshold": cal.threshold, "feature_bounds": bounds_dict}, indent=2)
    record_part = {
        "conformal_calibration": {
            "fp_budget_per_endpoint_day": cal.fp_budget_per_endpoint_day,
            "threshold": cal.threshold,
            "calibration_set_size": cal.calibration_set_size,
            "benign_baseline_rate": cal.benign_baseline_rate,
        },
        "feature_bounds": bounds_dict,
        "hyperparameters": _FOREST,
        "robustness_cards": [],
    }
    return onnx_bytes, metadata + "\n", record_part


def _assert_artifacts(out: Path, x: np.ndarray, feature_names: list[str], script: str) -> None:
    onnx_bytes, metadata, record_part = _reference(x, feature_names)
    assert (out / "model.onnx").read_bytes() == onnx_bytes
    assert (out / "model_metadata.json").read_text(encoding="utf-8") == metadata

    record = json.loads((out / "model_record.json").read_text(encoding="utf-8"))
    record["conformal_calibration"].pop("calibrated_at")
    assert record["training_script"] == script
    for key, expected in record_part.items():
        assert record[key] == expected, key


def _run(module, dataset: Path, out: Path) -> None:
    assert module.main(["--dataset", str(dataset), "--output-dir", str(out)]) == 0


@pytest.mark.parametrize(
    ("module", "platform", "rows"),
    [
        (
            train_linux,
            "linux",
            [{"argv": [f"/usr/bin/t{i}", "-x" * (i % 4 + 1), "y" * (i % 7)]} for i in range(60)],
        ),
        (
            train_windows,
            "windows",
            [{"cmdline": f"C:\\Windows\\t{i}.exe /{'q' * (i % 5)}"} for i in range(60)],
        ),
    ],
    ids=["linux", "windows"],
)
def test_the_cmdline_trainers_write_the_reference_artifacts(
    tmp_path: Path, module, platform: str, rows: list[dict]
) -> None:
    dataset = write_baseline_dataset(
        tmp_path / f"{platform}__dev__abc__2026-09-11",
        [json.dumps(r) for r in rows],
        capture_start=_START,
        capture_end=_END,
        platform=platform,
    )
    out = tmp_path / "out"
    _run(module, dataset, out)

    samples = unique_in_order(load_cmdlines(dataset, platform))
    x = np.array([cmdline.extract_features(s) for s in samples], dtype=np.float32)
    _assert_artifacts(out, x, list(cmdline.FEATURE_NAMES), module.TRAINING_SCRIPT)


def test_the_behavior_trainer_writes_the_reference_artifacts(tmp_path: Path) -> None:
    dataset = _write_behavior_dataset(tmp_path, _capture(40))
    out = tmp_path / "out"
    _run(train_behavior, dataset, out)

    x = train_behavior.load_matrix([dataset])
    _assert_artifacts(out, x, list(t1.FEATURE_NAMES), train_behavior.TRAINING_SCRIPT)
