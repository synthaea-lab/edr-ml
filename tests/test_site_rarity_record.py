"""The exact `extra` a site rarity run writes into its training record (issue #640).

The keys and their string formats are read by people and tools comparing records, so a
refactor of `train_site_rarity` must leave them as they are. One run with every option that
changes the record (scenarios, the global model) and one without them pin the whole dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from helpers import RobustnessStub, SiteRarityRun, cmdline_rows, robustness_card
from siblings import needs, sibling
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest

from synthaea_ml.registry.training_record import load_training_record
from synthaea_ml.training import train_site_rarity as rare

BEACON = sibling("scenarios", "scenarios", "beacon.yaml")
ONNX_OPSET = {"": 18, "ai.onnx.ml": 3}


def _global_model_dir(directory: Path) -> Path:
    forest = IsolationForest(n_estimators=50, random_state=42).fit(cmdline_rows(1000, seed=11))
    directory.mkdir(parents=True)
    (directory / "model.onnx").write_bytes(
        to_onnx(forest, cmdline_rows(1, seed=0), target_opset=ONNX_OPSET).SerializeToString()
    )
    (directory / "model_metadata.json").write_text(json.dumps({"threshold": -0.2}))
    return directory


def test_a_run_without_options_writes_exactly_these_extras(
    site_rarity_run: SiteRarityRun,
) -> None:
    extra = load_training_record(site_rarity_run("--withhold-fraction", "0.3")).extra

    assert extra == {
        "site": "acme",
        "rarity_k": "3",
        "rarity_min_hosts": "3",
        "snapshot_interval_hours": "24.0",
        "exec_events": "960",
        "rarity_known_fraction": "0.9500",
        "withhold_fraction": "0.3000",
        "fp_rate_test_cmdline_only": "0.0000",
        "fp_rate_test_with_rarity": "0.0000",
        "threshold_cmdline_only": "0.000000",
        "fp_rate_test_with_rarity_withheld": "0.0000",
        "rarity_floor": (
            "PASSED (withheld false-positive rate only: 0.00% against 0.00% + 1.00%; "
            "no scenarios given, so the escape rate was not checked)"
        ),
        "global_model_floor": (
            "NOT CHECKED against the global model: the floor above compares the candidate with "
            "this corpus's cmdline-only model; pass --global-model-dir to compare with the "
            "global model before this model leaves the lab"
        ),
    }


@needs("scenarios", "scenarios", "beacon.yaml")
def test_a_run_with_scenarios_and_the_global_model_writes_exactly_these_extras(
    tmp_path: Path, robustness_stub: RobustnessStub, site_rarity_run: SiteRarityRun
) -> None:
    robustness_stub(robustness_card(0.1, 0.0), robustness_card(0.1, 0.0))
    directory = _global_model_dir(tmp_path / "global")

    extra = load_training_record(
        site_rarity_run(
            "--withhold-fraction", "0.3", "--global-model-dir", str(directory),
            "--robustness-scenarios", str(BEACON),
        )
    ).extra

    assert extra == {
        "site": "acme",
        "rarity_k": "3",
        "rarity_min_hosts": "3",
        "snapshot_interval_hours": "24.0",
        "exec_events": "960",
        "rarity_known_fraction": "0.9500",
        "withhold_fraction": "0.3000",
        "fp_rate_test_cmdline_only": "0.0000",
        "fp_rate_test_with_rarity": "0.0000",
        "threshold_cmdline_only": "0.000000",
        "fp_rate_test_with_rarity_withheld": "0.0000",
        "rarity_floor": (
            "PASSED (withheld false-positive rate 0.00% against 0.00% + 1.00%; "
            "beacon: escape rate 10.00% vs 10.00% (cmdline-only model), "
            "median degradation 0.000 vs 0.000)"
        ),
        "global_model_floor": (
            "PASSED against the global model at its threshold -0.2 (model_metadata.json): "
            "withheld false-positive rate 0.00% against 0.00% + 1.00%; "
            "beacon: escape rate 10.00% vs 10.00% (global model), "
            "median degradation 0.000 vs 0.000"
        ),
        "fp_rate_test_global_model": "0.0000",
    }


def test_the_extras_format_each_measurement_to_its_fixed_precision() -> None:
    args = rare.SiteRarityArgs(
        site_corpus=Path("corpus.jsonl"), site_name="acme", output_dir=Path("out"), k=5,
        min_hosts=4, interval_hours=12.5, withhold_fraction=0.25,
        robustness_scenarios=(), global_model_dir=None,
    )
    matrices = rare.SiteMatrices(
        record_count=1200, cmdline=np.zeros((3, 9)), with_rarity=np.zeros((3, 11)),
        known_fraction=0.123456,
    )

    def measured(fp: float, threshold: float) -> rare.MeasuredForest:
        forest = SimpleNamespace(calibration=SimpleNamespace(threshold=threshold))
        return rare.MeasuredForest(forest, fp)

    floor = rare.FloorResult("cmdline-only model", 0.0125, 0.01875, [], [])
    shipped = rare.FloorResult(
        "global model", 0.03125, 0.01875, [], [],
        threshold=-0.0625, threshold_source="model_metadata.json",
    )

    extras = rare.record_extras(
        args, matrices, baseline=measured(0.0125, -0.1234567), candidate=measured(0.0064, -0.5),
        floor=floor, global_floor=shipped,
    )

    assert extras == {
        "site": "acme",
        "rarity_k": "5",
        "rarity_min_hosts": "4",
        "snapshot_interval_hours": "12.5",
        "exec_events": "3",
        "rarity_known_fraction": "0.1235",
        "withhold_fraction": "0.2500",
        "fp_rate_test_cmdline_only": "0.0125",
        "fp_rate_test_with_rarity": "0.0064",
        "threshold_cmdline_only": "-0.123457",
        "fp_rate_test_with_rarity_withheld": "0.0187",
        "rarity_floor": (
            "PASSED (withheld false-positive rate only: 1.88% against 1.25% + 1.00%; "
            "no scenarios given, so the escape rate was not checked)"
        ),
        "global_model_floor": (
            "PASSED against the global model at its threshold -0.0625 (model_metadata.json): "
            "withheld false-positive rate 1.88% against 3.12% + 1.00%; "
            "no scenarios given, so the escape rate was not checked"
        ),
        "fp_rate_test_global_model": "0.0312",
    }
