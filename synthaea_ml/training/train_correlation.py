"""Trains an Isolation Forest on the correlation vectors (2nd ML scorer) and exports it to ONNX.

SKELETON — waiting for the first real dataset: a `correlation_train.jsonl` produced by
`aggregate_correlation.py` from an `edr-cli capture-events` lab session
(~10-15 min of normal activity). See `docs/design/design-correlation-ml-vector.md`.

Scorer **distinct** from the cmdline scorer (`train_linux.py` / `train.py`): never the same
model file, never merged into a single feature space (decision of 2026-08-27). The model is
loaded on the Rust side by a future `CorrelationScorer` (not yet
written) then injected as an LLR term into the correlator's `update_belief()` (architecture to
be validated with Florian).

Choices to settle on real data (provisional values in `config.CORRELATION_FOREST`):
  - `contamination`: 0.01-0.02 anticipated — the behavioral counts of a normal process are
    close to zero and tightly clustered, the benign cluster should separate better than on the
    text features (which run at 0.05). To be confirmed on the real baseline.
  - threshold: conformal calibration (issue #46) — computes threshold from FP budget once
    real baselines are available.
  - agent gate: do not score a vector with `event_count < 3` (nearly empty window) — applied
    on the Rust side, not at training time.

Input (`--dataset`): a directory with `correlation_train.jsonl` (written by
`python -m synthaea_ml.data.aggregate_correlation --out <dir>/correlation_train.jsonl`) and its
`manifest.json` (`python -m synthaea_ml.data.manifest write --baseline-filename
correlation_train.jsonl ...`), verified before training.

Output (`--output-dir`, a registry version directory): `model.onnx`, `model_metadata.json`
(conformal threshold + feature bounds) and `model_record.json`.

Usage:
    python -m synthaea_ml.training.train_correlation \\
        --dataset ml/datasets/correlation/<name>/ \\
        --output-dir ml/registry/correlation-iforest-linux/0.1.0/

Deprecated: without options, the run reads `DATA_PATH` and writes `MODEL_PATH` and
`METADATA_PATH`, all inside the package source tree, with no manifest check and no model
record. That default will be removed in the next release.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthaea_ml.data.aggregate_correlation import LEGACY_OUT_PATH
from synthaea_ml.data.manifest import CORRELATION_DATASET_FILENAME
from synthaea_ml.features.correlation import FEATURE_NAMES
from synthaea_ml.registry.codec import write_training_record
from synthaea_ml.registry.dataset_binding import dataset_version_from_manifest
from synthaea_ml.registry.layout import METADATA_FILENAME, MODEL_FILENAME
from synthaea_ml.registry.records import DatasetVersion
from synthaea_ml.training.config import CALIBRATION, CORRELATION_FOREST
from synthaea_ml.training.pipeline import (
    export_onnx,
    fit_calibrated_forest,
    split_for_calibration,
    write_scorer_metadata,
)

TRAINING_SCRIPT = "synthaea_ml/training/train_correlation.py"

DATA_PATH = LEGACY_OUT_PATH
"""Deprecated: the dataset a run without options reads (`aggregate_correlation`'s default)."""
MODEL_PATH = Path(__file__).parent / "model_correlation.onnx"
"""Deprecated: where a run without options writes the model."""
METADATA_PATH = Path(__file__).parent / "model_correlation_metadata.json"
"""Deprecated: where a run without options writes the threshold and feature bounds."""

MIN_EVENT_COUNT_GATE = 3  # documentation for the Rust scorer, not applied here
EVENT_COUNT_IDX = FEATURE_NAMES.index("event_count")

# Sanity check — synthetic vectors, until we have real lab samples.
SANITY_CHECK_SAMPLES = {
    # spawn_count, connect_count, filewrite_count, uniq_daddr, uniq_dport,
    # has_full_chain, span_s, event_count
    "benign (quiet process)": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
    "benign (network client)": [1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 3.0],
    "suspicious (full dropper)": [1.0, 4.0, 3.0, 1.0, 2.0, 1.0, 1.5, 8.0],
    "suspicious (multi-dest scan)": [1.0, 20.0, 0.0, 15.0, 15.0, 0.0, 4.0, 21.0],
}


@dataclass(frozen=True)
class CorrelationRun:
    """Where a run reads its vectors and writes the model, its metadata and its record."""

    data_path: Path
    model_path: Path
    metadata_path: Path
    dataset_dir: Path | None
    """The verified dataset directory; `None` for the deprecated run, which has no
    manifest to verify and writes no model record."""

    @classmethod
    def from_dataset(cls, dataset_dir: Path, output_dir: Path) -> CorrelationRun:
        """A run on `dataset_dir`'s vectors that writes into the registry `output_dir`."""
        return cls(
            data_path=dataset_dir / CORRELATION_DATASET_FILENAME,
            model_path=output_dir / MODEL_FILENAME,
            metadata_path=output_dir / METADATA_FILENAME,
            dataset_dir=dataset_dir,
        )

    @classmethod
    def legacy(cls) -> CorrelationRun:
        """The deprecated run on the paths inside the package."""
        return cls(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            metadata_path=METADATA_PATH,
            dataset_dir=None,
        )


def load_dataset(path: Path | None = None) -> np.ndarray:
    """The vectors of the JSON-Lines file `path` (`DATA_PATH` when `None`), one row each."""
    path = DATA_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run a capture (`edr-cli capture-events`) "
            "then `python -m synthaea_ml.data.aggregate_correlation`."
        )
    vecs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            vecs.append(json.loads(line)["features"])
    if not vecs:
        raise ValueError(f"{path} is empty.")
    return np.array(vecs, dtype=np.float32)


def run(paths: CorrelationRun | None = None) -> None:
    """Train on `paths.data_path`, export the model and its metadata, write the model record
    when the run has a dataset directory, and print the sanity check.

    Without `paths`, the deprecated run on the paths inside the package.
    """
    paths = CorrelationRun.legacy() if paths is None else paths
    # Verify the manifest first: a mutated dataset is rejected before any model exists.
    dataset_version: DatasetVersion | None = None
    if paths.dataset_dir is not None:
        dataset_version = dataset_version_from_manifest(
            paths.dataset_dir, baseline_filename=CORRELATION_DATASET_FILENAME
        )

    X = load_dataset(paths.data_path)
    print(
        f"Training set: {X.shape[0]} vectors × {X.shape[1]} features ({', '.join(FEATURE_NAMES)})"
    )

    X_train, X_cal = split_for_calibration(X)
    print(f"Split: {len(X_train)} training, {len(X_cal)} calibration")

    forest = fit_calibrated_forest(X_train, X_cal, FEATURE_NAMES, CORRELATION_FOREST)
    clf = forest.model
    print(
        f"Conformal calibration: threshold={forest.calibration.threshold:.4f} "
        f"for ≤{CALIBRATION.fp_budget_per_endpoint_day} FP/endpoint/day"
    )
    print(f"Feature bounds computed with {CALIBRATION.feature_bounds_margin * 100}% margin")

    paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(clf, X_train, paths.model_path)
    print(f"Model exported → {paths.model_path}")
    write_scorer_metadata(paths.metadata_path, forest.calibration, forest.bounds)
    print(f"Metadata exported: {paths.metadata_path}")

    if dataset_version is not None:
        write_training_record(
            paths.model_path.parent,
            training_script=TRAINING_SCRIPT,
            dataset_versions=[dataset_version],
            hyperparameters=CORRELATION_FOREST.hyperparameters(),
            conformal_calibration=forest.calibration,
            feature_bounds=forest.bounds,
        )
        print(f"Model record written for dataset {dataset_version.name}")

    print("\nSanity check (scikit-learn):")
    for label, vec in SANITY_CHECK_SAMPLES.items():
        feats = np.array([vec], dtype=np.float32)
        score = clf.decision_function(feats)[0]
        verdict = "ANOMALY" if clf.predict(feats)[0] == -1 else "normal"
        print(f"  {label}: score={score:+.3f} -> {verdict}")

    print(
        f"\nRust scorer reminder: do not score if feature[{EVENT_COUNT_IDX}] "
        f"(event_count) < {MIN_EVENT_COUNT_GATE}."
    )


def parse_args(argv: Sequence[str] | None = None) -> CorrelationRun:
    """The run's paths from `argv` (the process's arguments when `None`).

    `--dataset` and `--output-dir` go together; without either, the deprecated run on the
    paths inside the package, with a `FutureWarning`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help=f"Directory with {CORRELATION_DATASET_FILENAME} + manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Registry version directory (model.onnx + model_record.json are written here).",
    )
    args = parser.parse_args(argv)
    if args.dataset is None and args.output_dir is None:
        warnings.warn(
            "train_correlation without --dataset/--output-dir reads and writes inside the "
            "package and writes no model record; pass --dataset and --output-dir. The "
            "default will be removed in the next release.",
            FutureWarning,
            stacklevel=2,
        )
        return CorrelationRun.legacy()
    if args.dataset is None or args.output_dir is None:
        parser.error("--dataset and --output-dir must be given together")
    return CorrelationRun.from_dataset(args.dataset, args.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Train from the command line `argv`; returns the exit status."""
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
