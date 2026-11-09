"""Trains a site-specific Isolation Forest model combining global + site corpus.

Site-specific model adaptation (Issue #49): Takes a global benign baseline + site-specific
telemetry corpus, trains a recalibrated model, validates against global-model-floor check
(using Issue #45's robustness evaluation), and outputs a site model ready for canary deployment.

Input contract:
- `--global-dataset`: Global baseline directory (same as train_linux.py input)
- `--site-corpus`: Site-specific corpus JSONL file (exported from control plane via /api/corpus/finalize)
- `--output-dir`: Registry version directory for site model

Site corpus format (JSONL):
```json
{"timestamp": "2026-09-23T10:00:00Z", "event_type": "exec", "event": {...}, "agent_id": "..."}
```

Global-model-floor check:
- Site model must match/exceed global model on shared scenario suite
- Uses run_robustness_evaluation() from Issue #45
- If floor check fails, training aborts

Usage:
    python -m synthaea_ml.training.train_site_model \\
        --global-dataset ml/datasets/baselines/linux__dev__abc__2026-09-11/ \\
        --site-corpus /path/to/corpus-v1.jsonl \\
        --output-dir ml/registry/cmdline-iforest-linux-site-acme/0.1.0/ \\
        --site-name acme-corp \\
        --robustness-scenarios scenarios/scenarios/beacon.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.ensemble import IsolationForest

from synthaea_ml.cli_logging import configure_cli_logging
from synthaea_ml.data.baselines import load_cmdlines, unique_in_order
from synthaea_ml.data.canonical import ml_cmdline_from_record
from synthaea_ml.data.manifest import DEFAULT_BASELINE_FILENAME
from synthaea_ml.evaluation.robustness import run_robustness_evaluation
from synthaea_ml.features.cmdline import extract_features
from synthaea_ml.hashing import sha256_file
from synthaea_ml.registry.codec import write_training_record
from synthaea_ml.registry.dataset_binding import dataset_version_from_manifest
from synthaea_ml.registry.layout import MODEL_FILENAME
from synthaea_ml.registry.records import DatasetVersion
from synthaea_ml.training.config import CMDLINE_FOREST, FLOOR_TOLERANCE, RANDOM_SEED
from synthaea_ml.training.floor import CardComparison, Scorer
from synthaea_ml.training.pipeline import export_onnx

logger = logging.getLogger(__name__)

TRAINING_SCRIPT = "synthaea_ml/training/train_site_model.py"
MIN_COMBINED_SAMPLES = 10


def load_global_baseline(baseline_dir: Path) -> list[str]:
    """Load global benign baseline (same as train_linux.py)."""
    baseline_path = baseline_dir / DEFAULT_BASELINE_FILENAME
    if not baseline_path.exists():
        raise FileNotFoundError(f"Global baseline not found: {baseline_path}")

    cmdlines = load_cmdlines(baseline_dir, "linux")
    unique = unique_in_order(cmdlines)

    logger.info("Global baseline: %d entries -> %d unique", len(cmdlines), len(unique))
    return unique


def load_site_corpus(corpus_path: Path) -> list[str]:
    """Load site-specific corpus from JSONL export."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Site corpus not found: {corpus_path}")

    cmdlines: list[str] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            # Site corpus format: {"event_type": "exec", "event": {...}}
            if record.get("event_type") == "exec" and "event" in record:
                cmdline = ml_cmdline_from_record(record["event"])
                cmdlines.append(cmdline)

    unique = unique_in_order(cmdlines)

    logger.info("Site corpus: %d entries -> %d unique", len(cmdlines), len(unique))
    return unique


def combine_corpora(global_corpus: list[str], site_corpus: list[str]) -> list[str]:
    """Combine global + site corpus with deduplication."""
    # Global samples first, then the site samples not already in it.
    combined = unique_in_order([*global_corpus, *site_corpus])

    logger.info(
        "Combined corpus: %d unique samples (%d global + %d site)",
        len(combined), len(global_corpus), len(site_corpus),
    )
    return combined


def load_global_model(path: Path) -> Scorer:
    """The global model the floor compares the site model with, unpickled from `path`.

    The pickle is a model the operator trained locally; it is never a downloaded artifact.
    Loading the shipped `model.onnx` instead, as `train_site_rarity` does, is a separate change.
    """
    with path.open("rb") as f:
        return cast(Scorer, pickle.load(f))  # noqa: S301 - the operator's own model, see above


def check_global_floor(site_model: Scorer, global_model: Scorer, scenario_yaml: Path) -> CardComparison:
    """The site model's robustness on `scenario_yaml` against the global model's.

    Both are evaluated with `run_robustness_evaluation` (#45) at tier T0 and the same mutation
    seed. The site model holds the floor when neither its escape rate nor its median score
    degradation exceeds the global model's plus `config.FLOOR_TOLERANCE`.
    """
    global_card = run_robustness_evaluation(
        model=global_model,
        scenario_yaml=scenario_yaml,
        tier="T0",
        mutation_seed=RANDOM_SEED,
    )
    site_card = run_robustness_evaluation(
        model=site_model,
        scenario_yaml=scenario_yaml,
        tier="T0",
        mutation_seed=RANDOM_SEED,
    )
    return CardComparison(candidate=site_card, reference=global_card)


def describe_global_floor(floor: CardComparison) -> list[str]:
    """The report lines of a global-model floor check: both models' metrics, the verdict, and
    each regression that failed it."""
    site, global_ = floor.candidate, floor.reference
    lines = [
        "",
        f"Global model: escape_rate={global_.escape_rate:.2%}, "
        f"median_degradation={global_.median_score_degradation:.3f}",
        f"Site model:   escape_rate={site.escape_rate:.2%}, "
        f"median_degradation={site.median_score_degradation:.3f}",
    ]
    if floor.passed:
        return [*lines, "Global-model-floor check PASSED"]
    lines.append("Global-model-floor check FAILED")
    if floor.escape_rate_regressed:
        lines.append(
            f"   - Escape rate regression: {site.escape_rate:.2%} > "
            f"{global_.escape_rate:.2%} + {FLOOR_TOLERANCE.escape_rate:.0%}"
        )
    if floor.median_degradation_regressed:
        lines.append(
            f"   - Median degradation regression: {site.median_score_degradation:.3f} > "
            f"{global_.median_score_degradation:.3f} + {FLOOR_TOLERANCE.median_degradation:.2f}"
        )
    return lines


@dataclass(frozen=True)
class SiteModelArgs:
    """The parsed command line of a run."""

    global_dataset: Path
    site_corpus: Path
    output_dir: Path
    site_name: str
    robustness_scenarios: Path | None
    global_model: Path | None


def parse_args(argv: Sequence[str] | None = None) -> SiteModelArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description="Train site-specific Isolation Forest model")
    parser.add_argument("--global-dataset", type=Path, required=True,
                        help="Global baseline directory (e.g., ml/datasets/baselines/linux__dev__abc__2026-09-11/)")
    parser.add_argument("--site-corpus", type=Path, required=True,
                        help="Site-specific corpus JSONL file (e.g., corpus-v1.jsonl)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for site model (e.g., ml/registry/cmdline-iforest-linux-site-acme/0.1.0/)")
    parser.add_argument("--site-name", type=str, required=True,
                        help="Site/tenant name for provenance")
    parser.add_argument("--robustness-scenarios", type=Path, default=None,
                        help="Scenario YAML for global-model-floor check (optional)")
    parser.add_argument("--global-model", type=Path, default=None,
                        help="Path to global model .pkl for floor check (optional)")
    args = parser.parse_args(argv)
    return SiteModelArgs(
        global_dataset=args.global_dataset,
        site_corpus=args.site_corpus,
        output_dir=args.output_dir,
        site_name=args.site_name,
        robustness_scenarios=args.robustness_scenarios,
        global_model=args.global_model,
    )


def train_site_forest(combined: list[str]) -> tuple[IsolationForest, np.ndarray]:
    """The forest fitted on the cmdline features of `combined`, and its feature matrix."""
    if len(combined) < MIN_COMBINED_SAMPLES:
        raise ValueError(
            f"Combined corpus has only {len(combined)} samples - need at least {MIN_COMBINED_SAMPLES}"
        )
    logger.info("=== Extracting features ===")
    X = np.array([extract_features(cmdline) for cmdline in combined], dtype=np.float32)
    logger.info("Feature matrix: %s", X.shape)

    logger.info("=== Training Isolation Forest ===")
    clf = IsolationForest(**CMDLINE_FOREST.hyperparameters())
    clf.fit(X)
    logger.info("Trained on %d samples", len(combined))
    return clf, X


def enforce_global_floor(args: SiteModelArgs, site_model: IsolationForest) -> None:
    """Run the global-model floor when both of its inputs are given and raise if it fails; warn
    and skip it when only one is."""
    if args.robustness_scenarios and args.global_model:
        print("\n=== Running global-model-floor check ===")
        floor = check_global_floor(
            site_model, load_global_model(args.global_model), args.robustness_scenarios
        )
        print("\n".join(describe_global_floor(floor)))
        if not floor.passed:
            raise ValueError("Global-model-floor check failed - site model does not meet baseline quality")
    elif args.robustness_scenarios or args.global_model:
        print("Warning: Skipping global-model-floor check (need both --robustness-scenarios and --global-model)")


def site_dataset_versions(args: SiteModelArgs, site_sample_count: int) -> list[DatasetVersion]:
    """The record's dataset versions: the global baseline's manifest, then the site corpus."""
    return [
        dataset_version_from_manifest(args.global_dataset),
        DatasetVersion(
            name=f"site-corpus-{args.site_name}",
            baseline_sha256=sha256_file(args.site_corpus),
            sample_count=site_sample_count,
        ),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    """Train, floor-check, export and record a site model from the command line `argv`."""
    args = parse_args(argv)
    configure_cli_logging()

    print("=== Loading corpora ===")
    global_corpus = load_global_baseline(args.global_dataset)
    site_corpus = load_site_corpus(args.site_corpus)
    combined = combine_corpora(global_corpus, site_corpus)

    clf, X = train_site_forest(combined)
    enforce_global_floor(args, clf)

    print("\n=== Exporting to ONNX ===")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / MODEL_FILENAME
    export_onnx(clf, X, onnx_path)
    print(f"Exported: {onnx_path}")

    print("\n=== Writing training record ===")
    write_training_record(
        args.output_dir,
        training_script=TRAINING_SCRIPT,
        dataset_versions=site_dataset_versions(args, len(site_corpus)),
        robustness_cards=[],  # Populated by separate robustness evaluation run
        scenario_replays=[],
    )

    print(f"\nSite model training complete: {args.output_dir}")
    print(f"   Global corpus: {len(global_corpus)} samples")
    print(f"   Site corpus: {len(site_corpus)} samples")
    print(f"   Combined: {len(combined)} samples")


if __name__ == "__main__":
    main()
