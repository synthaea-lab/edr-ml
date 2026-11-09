"""Re-verify a model record against on-disk state: the release gate's insertion point.

The gate does not retrain or replay; it checks that every dataset and scenario the
record binds still hashes to the value the record locked, and that each cached
``passed`` agrees with a re-computation of the passing criterion.
"""

from __future__ import annotations

from pathlib import Path

from synthaea_ml.data.manifest import (
    BEHAVIOR_CAPTURE_FILENAME,
    CORRELATION_DATASET_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    load_manifest,
    verify_manifest,
)
from synthaea_ml.hashing import sha256_file
from synthaea_ml.registry.codec import load_training_record
from synthaea_ml.registry.records import DatasetVersion, ScenarioReplayResult, compute_passed


def verify_training_record(
    model_dir: Path,
    baselines_root: Path,
    scenarios_root: Path,
) -> None:
    """Re-verify every ``DatasetVersion`` and every ``ScenarioReplayResult``
    in the model record against on-disk state.

    The release gate calls this before shipping a model. Two families of
    checks:

    * **Dataset provenance** (schema v1 semantics, unchanged): each dataset
      version's ``baseline_sha256`` and ``sample_count`` must still match the
      on-disk baseline's current manifest. Baseline moved → fail. Baseline
      mutated → fail.
    * **Scenario replay provenance** (schema v2, ADR-0009): each replay's
      ``scenario_yaml_sha256`` must still match ``scenarios_root/<name>.yaml``,
      and the recorded ``passed`` value must agree with a re-computation of
      the passing criterion (catches a mutated cache).

    Args:
        model_dir: The registry version directory.
        baselines_root: The parent directory holding baselines by name, e.g.
            ``ml/datasets/baselines/``. Each ``DatasetVersion.name`` is looked
            up as ``baselines_root/<name>/``.
        scenarios_root: The directory holding scenario yaml sidecars, e.g.
            ``<root>/scenarios/scenarios/``. Each ``ScenarioReplayResult.scenario_name`` is
            looked up as ``scenarios_root/<name>.yaml``.

    Raises:
        FileNotFoundError: If the model record itself, any referenced
            baseline directory or its manifest, or any referenced scenario
            yaml, is missing.
        ValueError: If a referenced baseline's current hash or sample count
            no longer matches what the record locked, or a referenced
            scenario yaml's current hash or a cached ``passed`` no longer
            matches.
    """
    record = load_training_record(model_dir)
    for dataset_version in record.dataset_versions:
        verify_dataset_version(dataset_version, baselines_root)
    for replay in record.scenario_replays:
        verify_scenario_replay(replay, scenarios_root)


def verify_dataset_version(dataset_version: DatasetVersion, baselines_root: Path) -> None:
    """Re-verify one ``DatasetVersion`` against ``baselines_root/<name>/``: its baseline must
    still match its manifest, and the manifest must still carry the hash and sample count
    the record locked.

    Raises:
        FileNotFoundError: If the baseline directory, its samples file or its manifest
            is missing.
        ValueError: If the baseline no longer matches its manifest, or the manifest's hash
            or sample count no longer matches the record.
    """
    baseline_dir = baselines_root / dataset_version.name
    if not baseline_dir.is_dir():
        raise FileNotFoundError(
            f"dataset {dataset_version.name!r} not found under {baselines_root}"
        )
    baseline_filename = _baseline_filename(baseline_dir)
    verify_manifest(baseline_dir, baseline_filename=baseline_filename)
    current = load_manifest(baseline_dir)
    if current.sample_sha256 != dataset_version.baseline_sha256:
        raise ValueError(
            f"dataset {dataset_version.name!r} hash mismatch: "
            f"training record locked {dataset_version.baseline_sha256}, "
            f"current baseline is {current.sample_sha256}"
        )
    if current.sample_count != dataset_version.sample_count:
        raise ValueError(
            f"dataset {dataset_version.name!r} sample_count mismatch: "
            f"training record locked {dataset_version.sample_count}, "
            f"current baseline has {current.sample_count}"
        )


def _baseline_filename(baseline_dir: Path) -> str:
    """The samples file inside ``baseline_dir``. A T1 behavior dataset is a raw agent
    capture named ``events.jsonl`` (#617), a correlation-scorer dataset is
    ``correlation_train.jsonl``; every other dataset is ``baseline.jsonl``.

    Raises:
        FileNotFoundError: If neither file is present.
    """
    candidates = (
        DEFAULT_BASELINE_FILENAME,
        BEHAVIOR_CAPTURE_FILENAME,
        CORRELATION_DATASET_FILENAME,
    )
    for filename in candidates:
        if (baseline_dir / filename).exists():
            return filename
    raise FileNotFoundError(
        f"baseline missing inside {baseline_dir}: expected {DEFAULT_BASELINE_FILENAME}, "
        f"{BEHAVIOR_CAPTURE_FILENAME} or {CORRELATION_DATASET_FILENAME}"
    )


def verify_scenario_replay(
    replay: ScenarioReplayResult,
    scenarios_root: Path,
) -> None:
    """Re-verify one ``ScenarioReplayResult`` against on-disk state.

    Two checks (ADR-0009):

    * ``scenario_yaml_sha256`` matches the current content hash of
      ``scenarios_root/<scenario_name>.yaml``.
    * Re-computing the passing criterion from the recorded
      ``expected_detections`` and ``observed_detections`` agrees with the
      cached ``passed`` field (a mutated cache is caught).

    The gate does *not* re-run the scenario; it verifies the record's
    self-consistency and its binding to the source yaml.

    Raises:
        FileNotFoundError: If the source yaml is missing.
        ValueError: If the yaml hash or the re-computed ``passed`` disagrees
            with the recorded value.
    """
    yaml_path = scenarios_root / f"{replay.scenario_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"scenario yaml {yaml_path} missing — cannot verify replay for "
            f"{replay.scenario_name!r}"
        )
    current_hash = sha256_file(yaml_path)
    if current_hash != replay.scenario_yaml_sha256:
        raise ValueError(
            f"scenario {replay.scenario_name!r} yaml hash mismatch: "
            f"replay locked {replay.scenario_yaml_sha256}, "
            f"current yaml is {current_hash}"
        )
    recomputed = compute_passed(replay.expected_detections, replay.observed_detections)
    if recomputed != replay.passed:
        raise ValueError(
            f"scenario {replay.scenario_name!r} passed field mutated: "
            f"cached {replay.passed}, re-computed {recomputed}"
        )

