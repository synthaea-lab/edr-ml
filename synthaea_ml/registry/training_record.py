"""Model record — the machine-parsable half of a model card.

Every registry entry has a human-readable ``card.md`` alongside its
``model.onnx`` (see ``ml/registry/README.md``). The card documents purpose,
metrics, FP rate, scenario-replay results and known limitations for a reader.

``model_record.json`` sits next to it, capturing exactly what a machine needs
to answer:

* "was this model trained on the datasets its card claims?" — the pre-existing
  binding, still enforced by hashing each baseline's content against
  ``sample_sha256``;
* "was this model replay-validated against the scenarios its card claims,
  and did those replays pass?" — the new binding introduced by ADR-0009,
  matched by hashing each scenario yaml against
  ``ScenarioReplayResult.scenario_yaml_sha256`` and by re-computing the
  ``passed`` field from the recorded counts.

Both are enforced at ``verify_training_record`` time and, through it, at
``verify_provenance`` time in the release gate.

This module is the stable import path for the record API. The implementation is
split by concern:

* ``synthaea_ml.registry.records`` — the dataclasses and the passing criterion;
* ``synthaea_ml.registry.codec`` — writing and typed reading of the JSON file;
* ``synthaea_ml.registry.dataset_binding`` — ``DatasetVersion`` from a manifest;
* ``synthaea_ml.registry.provenance`` — the release gate's re-verification.

Layout inside a version directory::

    ml/registry/<model>/<version>/
        model.onnx           # the artifact
        card.md              # for humans
        model_record.json    # for machines — this module writes/reads it

The record binds each ``DatasetVersion`` by its ``baseline_sha256`` (the
content hash the manifest already publishes), not by path — a dataset can move
on disk without invalidating the record, but a mutated baseline cannot pass
``verify_training_record`` even if it kept the same path. The same principle
applies to scenario yamls via ``scenario_yaml_sha256``.

Renamed from ``training.json`` in ADR-0009: the record now covers scenario
replay outcomes on top of dataset binding — no longer a "training" concept but
a full model-lifecycle record. Schema version bumped 1 → 2; readers reject v1
(matches the ``manifest.py`` pattern). No on-disk migration was required:
``training.json`` had never been materialised in the repo when this rename
landed (verified pre-ADR).
"""

from synthaea_ml.registry.codec import (
    RecordFormatError,
    load_training_record,
    record_from_dict,
    record_to_dict,
    write_training_record,
)
from synthaea_ml.registry.dataset_binding import (
    dataset_version_from_manifest,
    default_dataset_name,
)
from synthaea_ml.registry.layout import MODEL_RECORD_FILENAME
from synthaea_ml.registry.provenance import (
    verify_dataset_version,
    verify_robustness_card,
    verify_scenario_replay,
    verify_training_record,
)
from synthaea_ml.registry.records import (
    SCHEMA_VERSION,
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
    compute_passed,
)

__all__ = [
    "MODEL_RECORD_FILENAME",
    "SCHEMA_VERSION",
    "ConformalCalibration",
    "DatasetVersion",
    "Environment",
    "ExpectedDetection",
    "FeatureBounds",
    "MutationTestResult",
    "ObservedDetection",
    "RecordFormatError",
    "RobustnessCard",
    "ScenarioReplayResult",
    "TrainingRecord",
    "compute_passed",
    "dataset_version_from_manifest",
    "default_dataset_name",
    "load_training_record",
    "record_from_dict",
    "record_to_dict",
    "verify_dataset_version",
    "verify_robustness_card",
    "verify_scenario_replay",
    "verify_training_record",
    "write_training_record",
]
