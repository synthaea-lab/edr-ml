"""Read and write ``model_record.json``, the hashed on-disk form of a ``TrainingRecord``.

Writing packs every block field by field and serialises with
``json.dumps(indent=2, sort_keys=True) + "\\n"``: the release gate hashes these bytes, so
that format is a contract, not a style choice.

Reading checks the type of every value before it reaches a dataclass, so a malformed
record fails with a ``RecordFormatError`` naming the offending key rather than with a
``KeyError`` or ``TypeError`` from deep inside a constructor. Nested blocks must carry
exactly their dataclass's keys; unknown top-level keys are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from synthaea_ml.registry.layout import MODEL_RECORD_FILENAME
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
)
from synthaea_ml.timestamps import format_utc

READABLE_SCHEMA_VERSIONS = (2, SCHEMA_VERSION)
"""Schema v3 only added optional blocks (empty or absent by default), so v2 records load."""

_Block = TypeVar("_Block")
_Decoder = Callable[[object, str], _Block]
"""Decodes one nested block from its JSON value and its location in the record."""


class RecordFormatError(ValueError):
    """A ``model_record.json`` whose content does not match the schema: a missing or
    unexpected key, a value of the wrong JSON type, or an unknown schema version.

    A ``ValueError`` so that callers treating a ``ValueError`` as a failed provenance
    check, as the release gate does, reject a malformed record the same way.
    """


# ── Write ───────────────────────────────────────────────────────────────────


def record_to_dict(record: TrainingRecord) -> dict[str, object]:
    """The JSON object of ``record``. Explicit per-field packing, so a nested frozen
    dataclass with mutable-typed fields (list, dict) cannot leak a dataclass-internal
    representation into the JSON. The v3 calibration blocks are omitted when absent.
    """
    result: dict[str, object] = {
        "schema_version": record.schema_version,
        "trained_at": record.trained_at,
        "training_script": record.training_script,
        "dataset_versions": [asdict(v) for v in record.dataset_versions],
        "scenario_replays": [_scenario_replay_to_dict(r) for r in record.scenario_replays],
        "robustness_cards": [_robustness_card_to_dict(c) for c in record.robustness_cards],
        "hyperparameters": dict(record.hyperparameters),
        "extra": dict(record.extra),
    }
    if record.conformal_calibration is not None:
        result["conformal_calibration"] = asdict(record.conformal_calibration)
    if record.feature_bounds is not None:
        result["feature_bounds"] = asdict(record.feature_bounds)
    return result


def _scenario_replay_to_dict(replay: ScenarioReplayResult) -> dict[str, object]:
    return {
        "scenario_name": replay.scenario_name,
        "scenario_yaml_sha256": replay.scenario_yaml_sha256,
        "run_at": replay.run_at,
        "environment": asdict(replay.environment),
        "expected_detections": [asdict(e) for e in replay.expected_detections],
        "observed_detections": [asdict(o) for o in replay.observed_detections],
        "passed": replay.passed,
    }


def _robustness_card_to_dict(card: RobustnessCard) -> dict[str, object]:
    return {
        "scenario_name": card.scenario_name,
        "scenario_yaml_sha256": card.scenario_yaml_sha256,
        "tested_at": card.tested_at,
        "mutation_results": [asdict(m) for m in card.mutation_results],
        "escape_rate": card.escape_rate,
        "median_score_degradation": card.median_score_degradation,
        "worst_case_degradation": card.worst_case_degradation,
    }


def write_training_record(
    model_dir: Path,
    *,
    training_script: str,
    dataset_versions: list[DatasetVersion],
    scenario_replays: list[ScenarioReplayResult] | None = None,
    robustness_cards: list[RobustnessCard] | None = None,
    hyperparameters: dict[str, object] | None = None,
    trained_at: datetime | None = None,
    extra: dict[str, str] | None = None,
    conformal_calibration: ConformalCalibration | None = None,
    feature_bounds: FeatureBounds | None = None,
) -> TrainingRecord:
    """Write ``model_dir/model_record.json``.

    Args:
        model_dir: The registry version directory, e.g.
            ``ml/registry/cmdline-iforest-linux/0.2.0/``. Must already exist.
        training_script: Path-like string identifying the entry point,
            e.g. ``"synthaea_ml/training/train_linux.py"``. Recorded for
            traceability, not resolved.
        dataset_versions: One entry per baseline the run consumed. Callers
            build these via ``dataset_version_from_manifest`` so each has a
            re-verifiable hash before it reaches this function.
        scenario_replays: Zero or more replay outcomes to bind to this model
            version. Each is expected to have been produced by the
            replay engine which fills the yaml hash and cached ``passed`` in
            one shot. Empty list is legal (a model without replay validation
            is a model that will not ship under the strict release gate — a
            fact captured by ``verify_provenance``, not here).
        robustness_cards: Zero or more adversarial evaluation outcomes to bind
            to this model version. Each is produced by the robustness evaluation
            runner which fills the yaml hash and aggregated metrics. Empty list
            is legal (a model without robustness testing may not ship under
            strict release gate).
        hyperparameters: The knobs the run was configured with. Not
            interpreted here — round-trip only.
        trained_at: Timezone-aware datetime, normalised to UTC. Defaults
            to ``datetime.now(UTC)``.
        extra: Optional ``str`` → ``str`` metadata, namespaced under ``extra``
            so a future typed field cannot collide.
        conformal_calibration: Optional conformal prediction calibration metadata
            (issue #46). See ``synthaea_ml.calibration.calibrate_conformal``.
        feature_bounds: Optional per-feature bounds for OOD detection (issue #46).
            See ``synthaea_ml.calibration.compute_feature_bounds``.

    Returns:
        The ``TrainingRecord`` that was just written.

    Raises:
        FileNotFoundError: If ``model_dir`` does not exist.
        ValueError: If ``trained_at`` is naive, or ``dataset_versions`` is
            empty (a training run with no dataset would not lock anything).
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    if not dataset_versions:
        raise ValueError("dataset_versions must not be empty")
    if trained_at is None:
        trained_at = datetime.now(UTC)
    if trained_at.tzinfo is None:
        raise ValueError("trained_at must be timezone-aware")

    record = TrainingRecord(
        schema_version=SCHEMA_VERSION,
        trained_at=format_utc(trained_at),
        training_script=training_script,
        dataset_versions=list(dataset_versions),
        scenario_replays=list(scenario_replays) if scenario_replays else [],
        robustness_cards=list(robustness_cards) if robustness_cards else [],
        hyperparameters=dict(hyperparameters) if hyperparameters else {},
        extra=dict(extra) if extra else {},
        conformal_calibration=conformal_calibration,
        feature_bounds=feature_bounds,
    )

    (model_dir / MODEL_RECORD_FILENAME).write_text(
        json.dumps(record_to_dict(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


# ── Read ────────────────────────────────────────────────────────────────────


def load_training_record(model_dir: Path) -> TrainingRecord:
    """Read ``model_dir/model_record.json``.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not JSON (a ``ValueError``).
        RecordFormatError: If the schema version is not one this reader knows, or the
            content does not match the schema (a ``ValueError``).
    """
    path = model_dir / MODEL_RECORD_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"model record not found: {path}")
    return record_from_dict(json.loads(path.read_text(encoding="utf-8")))


def record_from_dict(payload: object) -> TrainingRecord:
    """The ``TrainingRecord`` of a decoded ``model_record.json`` object.

    Raises:
        RecordFormatError: If the schema version is unknown or a value does not match
            the schema.
    """
    record = _Fields(payload, "record")
    schema_version = record.optional("schema_version")
    if type(schema_version) is not int or schema_version not in READABLE_SCHEMA_VERSIONS:
        raise RecordFormatError(
            f"unsupported {MODEL_RECORD_FILENAME} schema_version: {schema_version!r} "
            f"(this reader knows 2 and {SCHEMA_VERSION})"
        )
    return TrainingRecord(
        schema_version=schema_version,
        trained_at=record.string("trained_at"),
        training_script=record.string("training_script"),
        dataset_versions=record.blocks("dataset_versions", _dataset_version),
        scenario_replays=record.blocks("scenario_replays", _scenario_replay, optional=True),
        robustness_cards=record.blocks("robustness_cards", _robustness_card, optional=True),
        hyperparameters=record.mapping("hyperparameters"),
        extra=record.string_mapping("extra"),
        conformal_calibration=record.optional_block(
            "conformal_calibration", _conformal_calibration
        ),
        feature_bounds=record.optional_block("feature_bounds", _feature_bounds),
    )


def _dataset_version(value: object, where: str) -> DatasetVersion:
    block = _Fields.of_dataclass(value, where, DatasetVersion)
    return DatasetVersion(
        name=block.string("name"),
        baseline_sha256=block.string("baseline_sha256"),
        sample_count=block.integer("sample_count"),
    )


def _conformal_calibration(value: object, where: str) -> ConformalCalibration:
    block = _Fields.of_dataclass(value, where, ConformalCalibration)
    return ConformalCalibration(
        fp_budget_per_endpoint_day=block.number("fp_budget_per_endpoint_day"),
        threshold=block.number("threshold"),
        calibration_set_size=block.integer("calibration_set_size"),
        benign_baseline_rate=block.number("benign_baseline_rate"),
        calibrated_at=block.string("calibrated_at"),
    )


def _feature_bounds(value: object, where: str) -> FeatureBounds:
    block = _Fields.of_dataclass(value, where, FeatureBounds)
    return FeatureBounds(
        feature_names=block.strings("feature_names"),
        min_values=block.numbers("min_values"),
        max_values=block.numbers("max_values"),
    )


def _scenario_replay(value: object, where: str) -> ScenarioReplayResult:
    block = _Fields.of_dataclass(value, where, ScenarioReplayResult)
    return ScenarioReplayResult(
        scenario_name=block.string("scenario_name"),
        scenario_yaml_sha256=block.string("scenario_yaml_sha256"),
        run_at=block.string("run_at"),
        environment=block.block("environment", _environment),
        expected_detections=block.blocks("expected_detections", _expected_detection),
        observed_detections=block.blocks("observed_detections", _observed_detection),
        passed=block.boolean("passed"),
    )


def _environment(value: object, where: str) -> Environment:
    block = _Fields.of_dataclass(value, where, Environment)
    return Environment(
        os=block.string("os"), kernel=block.string("kernel"), arch=block.string("arch")
    )


def _expected_detection(value: object, where: str) -> ExpectedDetection:
    block = _Fields.of_dataclass(value, where, ExpectedDetection)
    return ExpectedDetection(
        technique=block.string("technique"),
        rule=block.string("rule"),
        min_count=block.integer("min_count"),
        tolerance=block.integer("tolerance"),
    )


def _observed_detection(value: object, where: str) -> ObservedDetection:
    block = _Fields.of_dataclass(value, where, ObservedDetection)
    return ObservedDetection(
        technique=block.string("technique"),
        rule=block.string("rule"),
        count=block.integer("count"),
    )


def _robustness_card(value: object, where: str) -> RobustnessCard:
    block = _Fields.of_dataclass(value, where, RobustnessCard)
    return RobustnessCard(
        scenario_name=block.string("scenario_name"),
        scenario_yaml_sha256=block.string("scenario_yaml_sha256"),
        tested_at=block.string("tested_at"),
        mutation_results=block.blocks("mutation_results", _mutation_test_result),
        escape_rate=block.number("escape_rate"),
        median_score_degradation=block.number("median_score_degradation"),
        worst_case_degradation=block.number("worst_case_degradation"),
    )


def _mutation_test_result(value: object, where: str) -> MutationTestResult:
    block = _Fields.of_dataclass(value, where, MutationTestResult)
    return MutationTestResult(
        mutation_class=block.string("mutation_class"),
        intensity=block.string("intensity"),
        original_score=block.number("original_score"),
        mutated_score=block.number("mutated_score"),
        score_delta=block.number("score_delta"),
        threshold=block.number("threshold"),
        escaped=block.boolean("escaped"),
        seed=block.integer("seed"),
    )


# ── Typed field access ──────────────────────────────────────────────────────


class _Fields:
    """The keys of one JSON object of the record, each read as the type the schema gives
    it. ``where`` locates the object in error messages, e.g.
    ``record.scenario_replays[0].environment``.

    JSON has one number type, so a float field accepts an integer literal; ``true`` and
    ``false`` are never numbers, although Python's ``bool`` subclasses ``int``.
    """

    def __init__(self, value: object, where: str) -> None:
        if not isinstance(value, dict):
            raise RecordFormatError(f"{where}: expected a JSON object, got {_json_type(value)}")
        self._values: dict[str, object] = value
        self._where = where

    @classmethod
    def of_dataclass(cls, value: object, where: str, block_type: type) -> _Fields:
        """The fields of a nested block, which must carry exactly ``block_type``'s keys."""
        block = cls(value, where)
        expected = {f.name for f in fields(block_type)}
        missing = sorted(expected - block._values.keys())
        unexpected = sorted(block._values.keys() - expected)
        if missing or unexpected:
            raise RecordFormatError(
                f"{where}: {block_type.__name__} keys do not match the schema "
                f"(missing {missing}, unexpected {unexpected})"
            )
        return block

    def optional(self, key: str) -> object:
        """The raw value at ``key``, or ``None`` when absent."""
        return self._values.get(key)

    def string(self, key: str) -> str:
        """The string at ``key``."""
        value = self._required(key)
        if not isinstance(value, str):
            raise self._wrong_type(key, "a string", value)
        return value

    def integer(self, key: str) -> int:
        """The integer at ``key``."""
        value = self._required(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._wrong_type(key, "an integer", value)
        return value

    def number(self, key: str) -> float:
        """The number at ``key``."""
        return self._number(self._required(key), key)

    def boolean(self, key: str) -> bool:
        """The ``true``/``false`` at ``key``."""
        value = self._required(key)
        if not isinstance(value, bool):
            raise self._wrong_type(key, "a boolean", value)
        return value

    def strings(self, key: str) -> list[str]:
        """The array of strings at ``key``."""
        items = self._array(key, optional=False)
        if not all(isinstance(item, str) for item in items):
            raise self._wrong_type(key, "an array of strings", items)
        return [str(item) for item in items]

    def numbers(self, key: str) -> list[float]:
        """The array of numbers at ``key``."""
        return [
            self._number(item, f"{key}[{index}]")
            for index, item in enumerate(self._array(key, optional=False))
        ]

    def mapping(self, key: str) -> dict[str, object]:
        """The object at ``key`` as a plain dict, or an empty dict when absent."""
        if key not in self._values:
            return {}
        value = self._values[key]
        if not isinstance(value, dict):
            raise self._wrong_type(key, "a JSON object", value)
        return dict(value)

    def string_mapping(self, key: str) -> dict[str, str]:
        """The object of strings at ``key``, or an empty dict when absent."""
        mapping = self.mapping(key)
        if not all(isinstance(value, str) for value in mapping.values()):
            raise self._wrong_type(key, "an object of strings", mapping)
        return {name: str(value) for name, value in mapping.items()}

    def block(self, key: str, decode: _Decoder[_Block]) -> _Block:
        """The nested block at ``key``, decoded by ``decode``."""
        return decode(self._required(key), f"{self._where}.{key}")

    def optional_block(self, key: str, decode: _Decoder[_Block]) -> _Block | None:
        """The nested block at ``key``, or ``None`` when absent."""
        return self.block(key, decode) if key in self._values else None

    def blocks(
        self, key: str, decode: _Decoder[_Block], *, optional: bool = False
    ) -> list[_Block]:
        """The array of nested blocks at ``key``; an optional array may be absent (empty)."""
        return [
            decode(item, f"{self._where}.{key}[{index}]")
            for index, item in enumerate(self._array(key, optional=optional))
        ]

    def _array(self, key: str, *, optional: bool) -> list[object]:
        if optional and key not in self._values:
            return []
        value = self._required(key)
        if not isinstance(value, list):
            raise self._wrong_type(key, "an array", value)
        return value

    def _number(self, value: object, key: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._wrong_type(key, "a number", value)
        return value

    def _required(self, key: str) -> object:
        if key not in self._values:
            raise RecordFormatError(f"{self._where}: missing key {key!r}")
        return self._values[key]

    def _wrong_type(self, key: str, expected: str, value: object) -> RecordFormatError:
        return RecordFormatError(
            f"{self._where}.{key}: expected {expected}, got {_json_type(value)}"
        )



def _json_type(value: object) -> str:
    """The JSON name of a decoded value's type, for error messages."""
    match value:
        case None:
            return "null"
        case bool():
            return "a boolean"
        case int() | float():
            return "a number"
        case str():
            return "a string"
        case list():
            return "an array"
        case dict():
            return "an object"
        case _:
            return type(value).__name__
