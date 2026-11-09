"""Trains an Isolation Forest on the benign Windows baseline and exports it to ONNX.

**Windows-only model** (decision of 2026-08-27, see the Linux mirror train_linux.py's
docstring): one model per OS, no mixing Linux+Windows in a single score space
(statistical dilution rather than real calibration - see `docs/PLAN.md`). ML scores
remain an experimental complement anyway, never a final judge (see threat-model.md).

Input contract: a **baseline directory** (`--dataset`) containing:

- `baseline.jsonl`: one exec record per line, `{"cmdline": "C:\\Windows\\..."}` on
  Windows. ETW does not populate `argv[]`, so the flat cmdline string is used
  directly as one token.
- `manifest.json`: the sidecar written by `synthaea_ml.data.manifest` - see PR #170
  in the #44 stack.

The manifest is verified before training runs; a mutated baseline is rejected here,
not silently baked into a model card. After training, `model.onnx` and `training.json`
are written to `--output-dir` (a registry version directory). Rule 3 of
`ml/README.md` (a registry model can be rebuilt from its recorded dataset versions)
is enforced by these two writes together.

Usage:
    python -m synthaea_ml.training.train_windows \\
        --dataset ml/datasets/baselines/windows__desktop-user__abc__2026-09-11/ \\
        --output-dir ml/registry/cmdline-iforest-windows/0.2.0/
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from synthaea_ml.cli_logging import configure_cli_logging
from synthaea_ml.data.baselines import load_cmdlines, unique_in_order
from synthaea_ml.features.cmdline import FEATURE_NAMES, extract_features
from synthaea_ml.features.sanity_samples import WINDOWS_SANITY_SAMPLES
from synthaea_ml.registry.codec import write_training_record
from synthaea_ml.registry.dataset_binding import dataset_version_from_manifest
from synthaea_ml.registry.layout import METADATA_FILENAME, MODEL_FILENAME
from synthaea_ml.training.config import CALIBRATION, CMDLINE_FOREST
from synthaea_ml.training.pipeline import (
    describe_card,
    evaluate_robustness,
    export_onnx,
    fit_calibrated_forest,
    split_for_calibration,
    write_scorer_metadata,
)
from synthaea_ml.training.trainer_args import (
    CMDLINE_TRAINER_HELP,
    TrainerArgs,
    parse_trainer_args,
)

logger = logging.getLogger(__name__)

TRAINING_SCRIPT = "synthaea_ml/training/train_windows.py"

# The samples live in `features.sanity_samples`, shared with `export.verify_onnx`.
SANITY_CHECK_SAMPLES = WINDOWS_SANITY_SAMPLES


def load_baseline(baseline_dir: Path) -> list[str]:
    """Load `baseline_dir/baseline.jsonl` and return unique cmdlines.

    Windows baseline format is `{"cmdline": "C:\\\\Windows\\\\..."}` - the ETW-captured
    full image path, no argv splitting on this platform. Deduplication is by cmdline
    string: a process that runs 10 000 times adds one sample.
    """
    cmdlines = load_cmdlines(baseline_dir, "windows")
    unique = unique_in_order(cmdlines)
    logger.info(
        "Baseline: %d entries -> %d unique after deduplication", len(cmdlines), len(unique)
    )
    return unique


def _run_sanity_check(clf: IsolationForest) -> None:
    """Print the sanity-check scores. Not a gate - the ONNX parity check that
    `verify_onnx` runs against the shipped model is the real one - but a quick sniff
    test that the training loss did not collapse the boundary to a single sign."""
    print("\nSanity check (scikit-learn, before ONNX loading on the Rust side):")
    for label, cmdline in SANITY_CHECK_SAMPLES.items():
        feats = np.array([extract_features(cmdline)], dtype=np.float32)
        score = clf.decision_function(feats)[0]
        pred = clf.predict(feats)[0]
        verdict = "ANOMALY" if pred == -1 else "normal"
        print(f"  {label}: score={score:+.3f} -> {verdict}")


def parse_args(argv: Sequence[str] | None = None) -> TrainerArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    return parse_trainer_args(argv, description=__doc__, help_text=CMDLINE_TRAINER_HELP)


def run(args: TrainerArgs) -> None:
    """Train, export and record the model the options `args` describe."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Verify each manifest before touching anything else. A mutated baseline is
    # rejected here (raises ValueError from verify_manifest) rather than silently
    # producing a model that points at a dataset version it was not trained on.
    dataset_versions = [dataset_version_from_manifest(d) for d in args.datasets]

    cmdlines: list[str] = []
    for d in args.datasets:
        cmdlines.extend(load_baseline(d))
    # Dedupe once across the concatenated samples so a command line seen in two
    # captures does not double-weight the boundary.
    cmdlines = unique_in_order(cmdlines)
    print(
        f"Concatenated & re-deduped across {len(args.datasets)} dataset(s): "
        f"{len(cmdlines)} unique samples"
    )

    X = np.array([extract_features(c) for c in cmdlines], dtype=np.float32)
    X_train, X_cal = split_for_calibration(X)
    print(f"Split: {len(X_train)} training, {len(X_cal)} calibration")

    forest = fit_calibrated_forest(X_train, X_cal, FEATURE_NAMES, CMDLINE_FOREST)
    print(
        f"Conformal calibration: threshold={forest.calibration.threshold:.4f} "
        f"for ≤{CALIBRATION.fp_budget_per_endpoint_day} FP/endpoint/day"
    )
    print(f"Feature bounds computed with {CALIBRATION.feature_bounds_margin * 100}% margin")

    export_onnx(forest.model, X_train, args.output_dir / MODEL_FILENAME)
    metadata_path = args.output_dir / METADATA_FILENAME
    write_scorer_metadata(metadata_path, forest.calibration, forest.bounds)
    print(f"Metadata exported: {metadata_path}")

    if args.robustness_scenarios:
        print(f"\nRunning robustness evaluation on {len(args.robustness_scenarios)} scenario(s)...")
        if args.robustness_events:
            print(f"  Using events from: {args.robustness_events}")
    robustness_cards = evaluate_robustness(
        forest.model, args.robustness_scenarios, args.robustness_events, tier="T0"
    )
    for card in robustness_cards:
        print(describe_card(card))

    write_training_record(
        args.output_dir,
        training_script=TRAINING_SCRIPT,
        dataset_versions=dataset_versions,
        robustness_cards=robustness_cards,
        hyperparameters=CMDLINE_FOREST.hyperparameters(),
        conformal_calibration=forest.calibration,
        feature_bounds=forest.bounds,
    )

    print(f"Model trained on {len(cmdlines)} benign examples -> {args.output_dir}")

    _run_sanity_check(forest.model)


def main(argv: Sequence[str] | None = None) -> int:
    """Train from the command line `argv`; returns the exit status."""
    args = parse_args(argv)
    configure_cli_logging()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
