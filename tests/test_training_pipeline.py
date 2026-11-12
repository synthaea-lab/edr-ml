"""The shared trainer steps in `training/pipeline.py`.

The bytes they produce end to end are pinned in `test_trainer_artifacts.py`; these tests
cover the contract of each step on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from helpers import cmdline_rows

from synthaea_ml.features import cmdline
from synthaea_ml.training import pipeline
from synthaea_ml.training.config import CMDLINE_FOREST

NAMES = list(cmdline.FEATURE_NAMES)


def _forest() -> pipeline.CalibratedForest:
    x_train, x_cal = pipeline.split_for_calibration(cmdline_rows(50, seed=1))
    return pipeline.fit_calibrated_forest(x_train, x_cal, NAMES, CMDLINE_FOREST)


def test_the_split_holds_out_thirty_percent_and_is_seeded() -> None:
    x = cmdline_rows(50, seed=1)
    x_train, x_cal = pipeline.split_for_calibration(x)
    assert (len(x_train), len(x_cal)) == (35, 15)
    again_train, _ = pipeline.split_for_calibration(x)
    assert np.array_equal(x_train, again_train)


def test_the_calibrated_forest_carries_the_threshold_and_the_training_bounds() -> None:
    forest = _forest()
    assert forest.model.get_params()["contamination"] == CMDLINE_FOREST.contamination
    assert forest.calibration.calibration_set_size == 15
    assert forest.bounds.feature_names == NAMES
    assert len(forest.bounds.min_values) == len(NAMES)


def test_the_onnx_export_uses_the_pinned_opsets(tmp_path: Path) -> None:
    forest = _forest()
    path = tmp_path / "model.onnx"
    pipeline.export_onnx(forest.model, cmdline_rows(3, seed=2), path)
    opsets = {o.domain: o.version for o in onnx.load(str(path)).opset_import}
    assert opsets["ai.onnx.ml"] == pipeline.ONNX_TARGET_OPSET["ai.onnx.ml"]
    assert opsets[""] == pipeline.ONNX_TARGET_OPSET[""]


def test_the_scorer_metadata_is_the_threshold_and_the_bounds(tmp_path: Path) -> None:
    forest = _forest()
    path = tmp_path / "model_metadata.json"
    pipeline.write_scorer_metadata(path, forest.calibration, forest.bounds)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text) == {
        "threshold": forest.calibration.threshold,
        "feature_bounds": {
            "feature_names": NAMES,
            "min_values": forest.bounds.min_values,
            "max_values": forest.bounds.max_values,
        },
    }


def test_the_scorer_metadata_has_lf_line_endings_on_every_platform(tmp_path: Path) -> None:
    forest = _forest()
    path = tmp_path / "model_metadata.json"

    pipeline.write_scorer_metadata(path, forest.calibration, forest.bounds)

    assert b"\r" not in path.read_bytes()


def test_robustness_runs_every_scenario_in_order_with_the_seed_and_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return kwargs["scenario_yaml"].stem

    monkeypatch.setattr(pipeline, "run_robustness_evaluation", fake)
    events = Path("events.jsonl")
    cards = pipeline.evaluate_robustness(
        _forest(), [Path("a.yaml"), Path("b.yaml")], events, tier="T1"
    )
    assert cards == ["a", "b"]
    assert [(c["tier"], c["mutation_seed"], c["events_source"]) for c in calls] == [
        ("T1", 42, events)
    ] * 2


def test_robustness_scores_the_forest_at_its_calibrated_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        pipeline, "run_robustness_evaluation", lambda **kwargs: calls.append(kwargs)
    )
    forest = _forest()
    pipeline.evaluate_robustness(forest, [Path("beacon.yaml")], None, tier="T0")
    assert [(c["model"], c["threshold"]) for c in calls] == [
        (forest.model, forest.calibration.threshold)
    ]


def test_a_failed_robustness_run_propagates_and_names_the_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pipeline, "run_robustness_evaluation", fake)
    with pytest.raises(ValueError, match="boom") as raised:
        pipeline.evaluate_robustness(_forest(), [Path("beacon.yaml")], None, tier="T0")
    assert raised.value.__notes__ == ["robustness evaluation failed for beacon.yaml"]


def test_no_scenarios_means_no_cards() -> None:
    assert pipeline.evaluate_robustness(_forest(), [], None, tier="T0") == []
