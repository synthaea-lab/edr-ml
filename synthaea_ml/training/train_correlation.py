"""Trains an Isolation Forest on the correlation vectors (2nd ML scorer) and exports it to ONNX.

SKELETON — waiting for the first real dataset: `ml/data/correlation_train.jsonl`,
produced by `aggregate_correlation.py` from an `edr-cli capture-events` lab session
(~10-15 min of normal activity). See `docs/design/design-correlation-ml-vector.md`.

Scorer **distinct** from the cmdline scorer (`train_linux.py` / `train.py`): never the same
model file, never merged into a single feature space (decision of 2026-08-27). Output:
`ml/model_correlation.onnx`, loaded on the Rust side by a future `CorrelationScorer` (not yet
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

Usage: `python -m synthaea_ml.training.train_correlation`  (venv `ml/.venv`: scikit-learn /
skl2onnx / onnx).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from synthaea_ml.features.correlation import FEATURE_NAMES
from synthaea_ml.training.config import CALIBRATION, CORRELATION_FOREST
from synthaea_ml.training.pipeline import (
    export_onnx,
    fit_calibrated_forest,
    split_for_calibration,
    write_scorer_metadata,
)

DATA_PATH = Path(__file__).parent / "data" / "correlation_train.jsonl"
MODEL_PATH = Path(__file__).parent / "model_correlation.onnx"
METADATA_PATH = Path(__file__).parent / "model_correlation_metadata.json"

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


def load_dataset() -> np.ndarray:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found — run a capture (`edr-cli capture-events`) "
            "then `python -m synthaea_ml.data.aggregate_correlation`."
        )
    vecs = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            vecs.append(json.loads(line)["features"])
    if not vecs:
        raise ValueError(f"{DATA_PATH} is empty.")
    return np.array(vecs, dtype=np.float32)


def run() -> None:
    """Train on `DATA_PATH`, export the model and its metadata, and print the sanity check."""
    X = load_dataset()
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

    export_onnx(clf, X_train, MODEL_PATH)
    print(f"Model exported → {MODEL_PATH}")
    write_scorer_metadata(METADATA_PATH, forest.calibration, forest.bounds)
    print(f"Metadata exported: {METADATA_PATH}")

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


def main(argv: Sequence[str] | None = None) -> int:
    """Train from the command line `argv`, which takes no option but `--help`; returns the
    exit status."""
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
