"""A malformed ``model_record.json`` is rejected with a ``RecordFormatError`` that names
the offending key, never with a ``KeyError`` or ``TypeError`` from a constructor, and is
still a ``ValueError`` so the release gate treats it as a failed provenance check.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from test_training_record_golden import GOLDEN_V3_RECORD

from synthaea_ml.registry.codec import RecordFormatError, record_from_dict, record_to_dict
from synthaea_ml.registry.training_record import MODEL_RECORD_FILENAME, load_training_record

Payload = dict[str, object]


def _golden_payload() -> Payload:
    payload: Payload = json.loads(GOLDEN_V3_RECORD)
    return payload


def _replay(payload: Payload) -> dict[str, object]:
    replays = payload["scenario_replays"]
    assert isinstance(replays, list)
    replay: dict[str, object] = replays[0]
    return replay


def _card(payload: Payload) -> dict[str, object]:
    cards = payload["robustness_cards"]
    assert isinstance(cards, list)
    card: dict[str, object] = cards[0]
    return card


def _mutation(payload: Payload) -> dict[str, object]:
    results = _card(payload)["mutation_results"]
    assert isinstance(results, list)
    result: dict[str, object] = results[0]
    return result


def _bounds(payload: Payload) -> dict[str, object]:
    bounds = payload["feature_bounds"]
    assert isinstance(bounds, dict)
    return bounds


_MALFORMATIONS: dict[str, tuple[Callable[[Payload], None], str]] = {
    "missing trained_at": (lambda p: p.pop("trained_at"), "record: missing key 'trained_at'"),
    "training_script is a number": (
        lambda p: p.update(training_script=3),
        "record.training_script: expected a string, got a number",
    ),
    "dataset_versions is null": (
        lambda p: p.update(dataset_versions=None),
        "record.dataset_versions: expected an array, got null",
    ),
    "sample_count is a boolean": (
        lambda p: p.update(
            dataset_versions=[{"name": "n", "baseline_sha256": "0", "sample_count": True}]
        ),
        "record.dataset_versions[0].sample_count: expected an integer, got a boolean",
    ),
    "dataset version with an unexpected key": (
        lambda p: p.update(
            dataset_versions=[
                {"name": "n", "baseline_sha256": "0", "sample_count": 1, "path": "/x"}
            ]
        ),
        "record.dataset_versions[0]: DatasetVersion keys do not match the schema "
        "(missing [], unexpected ['path'])",
    ),
    "replay environment missing arch": (
        lambda p: _replay(p).update(environment={"os": "linux", "kernel": "6.6"}),
        "record.scenario_replays[0].environment: Environment keys do not match the schema "
        "(missing ['arch'], unexpected [])",
    ),
    "replay passed is a string": (
        lambda p: _replay(p).update(passed="true"),
        "record.scenario_replays[0].passed: expected a boolean, got a string",
    ),
    "mutation score is a string": (
        lambda p: _mutation(p).update(mutated_score="-0.1"),
        "record.robustness_cards[0].mutation_results[0].mutated_score: "
        "expected a number, got a string",
    ),
    "feature bound is null": (
        lambda p: _bounds(p).update(min_values=[0.0, None]),
        "record.feature_bounds.min_values[1]: expected a number, got null",
    ),
    "feature name is a number": (
        lambda p: _bounds(p).update(feature_names=["a", 1]),
        "record.feature_bounds.feature_names: expected an array of strings, got an array",
    ),
    "calibration is null": (
        lambda p: p.update(conformal_calibration=None),
        "record.conformal_calibration: expected a JSON object, got null",
    ),
    "extra value is a number": (
        lambda p: p.update(extra={"commit_sha": 1}),
        "record.extra: expected an object of strings, got an object",
    ),
    "hyperparameters is an array": (
        lambda p: p.update(hyperparameters=[]),
        "record.hyperparameters: expected a JSON object, got an array",
    ),
}


@pytest.mark.parametrize(
    ("malform", "message"), list(_MALFORMATIONS.values()), ids=list(_MALFORMATIONS)
)
def test_malformed_record_raises_record_format_error_naming_the_key(
    malform: Callable[[Payload], None], message: str
) -> None:
    payload = _golden_payload()
    malform(payload)

    with pytest.raises(RecordFormatError) as raised:
        record_from_dict(payload)

    assert message in str(raised.value)


def test_record_that_is_not_an_object_raises_record_format_error() -> None:
    with pytest.raises(RecordFormatError, match="record: expected a JSON object, got an array"):
        record_from_dict([])


def test_record_format_error_is_a_value_error(tmp_path: Path) -> None:
    """The release gate maps ``ValueError`` to a failed provenance check."""
    payload = _golden_payload()
    payload.pop("dataset_versions")
    (tmp_path / MODEL_RECORD_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing key 'dataset_versions'"):
        load_training_record(tmp_path)


@pytest.mark.parametrize("schema_version", [None, 1, 4, "3", 3.0, True])
def test_unknown_schema_version_raises_record_format_error(schema_version: object) -> None:
    payload = _golden_payload()
    payload["schema_version"] = schema_version

    with pytest.raises(RecordFormatError, match="unsupported model_record.json schema_version"):
        record_from_dict(payload)


def test_unknown_top_level_key_is_ignored() -> None:
    payload = _golden_payload()
    expected = record_from_dict(copy.deepcopy(payload))
    payload["written_by"] = "a future writer"

    assert record_from_dict(payload) == expected


def test_float_field_accepts_an_integer_literal() -> None:
    payload = _golden_payload()
    _card(payload)["escape_rate"] = 0

    assert record_from_dict(payload).robustness_cards[0].escape_rate == 0


def test_decoding_the_encoded_golden_returns_the_same_object() -> None:
    payload = _golden_payload()

    assert record_to_dict(record_from_dict(payload)) == payload
