"""Trains the T1 behavior Isolation Forest on the 17 cmdline + correlation features (issue #617).

T1 scores one process incarnation at a time on cmdline (9) + correlation (8) features
(`synthaea_ml.features.t1`; the lineage is judged by rules, ADR-0034). Unlike the T0
trainers it cannot learn from a `baseline.jsonl` of command lines: the correlation block
needs pids and process generations, so its input is a raw agent capture of normal activity.

Input contract: a dataset directory (`--dataset`) containing

- `events.jsonl`: the agent's own capture (`crates/schema::Event` JSON, identity nested
  under `meta`), recorded on a host doing normal work.
- `manifest.json`: written by `python -m synthaea_ml.data.manifest` with
  `--baseline-filename events.jsonl`; verified before training, as for the T0 trainers.

Output (`--output-dir`, a registry version directory): `model.onnx`,
`model_metadata.json` (conformal threshold + feature bounds) and `model_record.json`.

What this does **not** do: ship a model. The agent does not load `ml::BehaviorScorer` yet,
and the benign corpus must be a real capture. Run `--robustness-scenarios` to record the
T1 robustness card (command-line mutators; the parent is not scored, so no lineage mutator).

Usage:
    python -m synthaea_ml.training.train_behavior \\
        --dataset ml/datasets/baselines/linux__dev__abc__2026-10-02/ \\
        --output-dir ml/registry/behavior-iforest-linux/0.1.0/
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from synthaea_ml.cli_logging import configure_cli_logging
from synthaea_ml.data.behavior_dataset import load_records
from synthaea_ml.data.manifest import BEHAVIOR_CAPTURE_FILENAME
from synthaea_ml.features import t1
from synthaea_ml.registry.codec import write_training_record
from synthaea_ml.registry.dataset_binding import dataset_version_from_manifest
from synthaea_ml.registry.layout import METADATA_FILENAME, MODEL_FILENAME
from synthaea_ml.training.config import BEHAVIOR_FOREST, BEHAVIOR_MIN_SAMPLES, CALIBRATION
from synthaea_ml.training.pipeline import (
    describe_card,
    evaluate_robustness,
    export_onnx,
    fit_calibrated_forest,
    split_for_calibration,
    write_scorer_metadata,
)
from synthaea_ml.training.trainer_args import TrainerArgs, TrainerHelp, parse_trainer_args

logger = logging.getLogger(__name__)

TRAINING_SCRIPT = "synthaea_ml/training/train_behavior.py"


def load_matrix(dataset_dirs: Sequence[Path]) -> np.ndarray:
    """The deduplicated 17-feature matrix of every dataset's capture.

    Deduplicated by feature vector: an identical process behavior seen 10 000 times adds
    one sample, so the boundary is not dragged toward the most repeated activity.
    """
    rows: dict[tuple[float, ...], None] = {}
    total = 0
    for d in dataset_dirs:
        for record in load_records(d / BEHAVIOR_CAPTURE_FILENAME):
            total += 1
            rows[tuple(t1.extract_features(record))] = None
    logger.info(
        "Behavior capture: %d process incarnations -> %d distinct vectors", total, len(rows)
    )
    if len(rows) < BEHAVIOR_MIN_SAMPLES:
        raise ValueError(
            f"only {len(rows)} distinct behavior vectors (need at least {BEHAVIOR_MIN_SAMPLES}): "
            "capture a longer normal-activity session"
        )
    return np.array(list(rows), dtype=np.float32)


BEHAVIOR_TRAINER_HELP = TrainerHelp(
    dataset="One or more directories, each with events.jsonl + manifest.json.",
    output_dir="Registry version directory (model.onnx + model_record.json are written here).",
    robustness_scenarios=(
        "Scenario yamls for T1 adversarial robustness evaluation (issue #45/#617)."
    ),
    robustness_events=(
        "Capture of MALICIOUS activity matching the scenarios (events.jsonl, agent wire "
        "format). Without it the evaluation falls back to synthetic events."
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> TrainerArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    return parse_trainer_args(argv, description=__doc__, help_text=BEHAVIOR_TRAINER_HELP)


def run(args: TrainerArgs) -> None:
    """Train, export and record the T1 model the options `args` describe."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Verify every manifest first: a mutated capture is rejected before any model exists.
    dataset_versions = [
        dataset_version_from_manifest(d, baseline_filename=BEHAVIOR_CAPTURE_FILENAME)
        for d in args.datasets
    ]

    X = load_matrix(args.datasets)
    X_train, X_cal = split_for_calibration(X)
    print(f"Split: {len(X_train)} training, {len(X_cal)} calibration")

    forest = fit_calibrated_forest(X_train, X_cal, t1.FEATURE_NAMES, BEHAVIOR_FOREST)
    print(
        f"Conformal calibration: threshold={forest.calibration.threshold:.4f} "
        f"for <={CALIBRATION.fp_budget_per_endpoint_day} FP/endpoint/day"
    )

    export_onnx(forest.model, X_train, args.output_dir / MODEL_FILENAME)
    write_scorer_metadata(
        args.output_dir / METADATA_FILENAME, forest.calibration, forest.bounds
    )

    robustness_cards = evaluate_robustness(
        forest.model, args.robustness_scenarios, args.robustness_events, tier="T1"
    )
    for card in robustness_cards:
        print(describe_card(card))

    write_training_record(
        args.output_dir,
        training_script=TRAINING_SCRIPT,
        dataset_versions=dataset_versions,
        robustness_cards=robustness_cards,
        hyperparameters=BEHAVIOR_FOREST.hyperparameters(),
        conformal_calibration=forest.calibration,
        feature_bounds=forest.bounds,
    )
    print(f"T1 model trained on {len(X)} distinct behavior vectors -> {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    """Train from the command line `argv`; returns the exit status."""
    args = parse_args(argv)
    configure_cli_logging()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
