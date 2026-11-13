"""Tests for `synthaea_ml.evaluation.replay_cli` — the command that binds scenario
replays to a model record (issue #49).

The properties that need to hold:

- Replaying a passing scenario writes one `ScenarioReplayResult` into the record, with the
  scenario's yaml hash and `passed = True`, and returns exit status 0.
- A failing scenario is still written, and the command returns 1.
- `--append` keeps the record's existing replays; without it they are replaced.
- Every other field of the record (trained_at, dataset_versions, ...) survives untouched.
"""

from __future__ import annotations

from pathlib import Path

from synthaea_ml.evaluation.replay_cli import main
from synthaea_ml.registry.codec import load_training_record, write_training_record
from synthaea_ml.registry.records import DatasetVersion, ScenarioReplayResult

_SCENARIO_YAML = """\
name: test-beacon
platform: linux
script: beacon.sh
simulates: >
  A test scenario.
expected_detections:
  - technique: "T1071/T1041"
    rule: check_beacon
    min_count: 1
    tolerance: 0
notes: >
  Test fixture.
"""


def _model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_training_record(
        model_dir,
        training_script="tests/test_replay_cli.py",
        dataset_versions=[DatasetVersion(name="x", baseline_sha256="0" * 64, sample_count=1)],
    )
    return model_dir


def _scenario(tmp_path: Path) -> Path:
    p = tmp_path / "test-beacon.yaml"
    p.write_text(_SCENARIO_YAML, encoding="utf-8")
    return p


def _alerts(tmp_path: Path, techniques: list[str]) -> Path:
    p = tmp_path / "alerts.ndjson"
    lines = [f'{{"timestamp_ns":1,"technique":"{t}","message":"m"}}' for t in techniques]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def test_a_passing_replay_is_written_with_its_hash_and_returns_zero(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    scenario = _scenario(tmp_path)
    alerts = _alerts(tmp_path, ["T1071/T1041"])

    rc = main(["--model-dir", str(model_dir), "--replay", str(scenario), str(alerts)])

    assert rc == 0
    record = load_training_record(model_dir)
    assert len(record.scenario_replays) == 1
    replay = record.scenario_replays[0]
    assert replay.scenario_name == "test-beacon"
    assert replay.passed is True
    assert replay.observed_detections[0].technique == "T1071/T1041"
    # The yaml hash really binds the sidecar's bytes.
    import hashlib

    assert replay.scenario_yaml_sha256 == hashlib.sha256(scenario.read_bytes()).hexdigest()


def test_a_failing_replay_is_written_and_returns_one(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    scenario = _scenario(tmp_path)
    alerts = _alerts(tmp_path, [])  # the expected technique never fired

    rc = main(["--model-dir", str(model_dir), "--replay", str(scenario), str(alerts)])

    assert rc == 1
    record = load_training_record(model_dir)
    assert len(record.scenario_replays) == 1
    assert record.scenario_replays[0].passed is False


def test_without_append_the_replays_are_replaced(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    # Seed the record with a prior replay.
    seeded = load_training_record(model_dir)
    prior = ScenarioReplayResult(
        scenario_name="old",
        scenario_yaml_sha256="0" * 64,
        run_at="2026-01-01T00:00:00Z",
        environment=seeded.scenario_replays[0].environment
        if seeded.scenario_replays
        else _env(),
        expected_detections=[],
        observed_detections=[],
        passed=True,
    )
    from dataclasses import replace

    from synthaea_ml.registry.codec import write_record

    write_record(model_dir, replace(seeded, scenario_replays=[prior]))

    scenario = _scenario(tmp_path)
    alerts = _alerts(tmp_path, ["T1071/T1041"])
    main(["--model-dir", str(model_dir), "--replay", str(scenario), str(alerts)])

    names = [r.scenario_name for r in load_training_record(model_dir).scenario_replays]
    assert names == ["test-beacon"]


def test_append_keeps_the_existing_replays(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    scenario = _scenario(tmp_path)
    alerts = _alerts(tmp_path, ["T1071/T1041"])

    main(["--model-dir", str(model_dir), "--replay", str(scenario), str(alerts)])
    main(["--model-dir", str(model_dir), "--replay", str(scenario), str(alerts), "--append"])

    names = [r.scenario_name for r in load_training_record(model_dir).scenario_replays]
    assert names == ["test-beacon", "test-beacon"]


def test_the_rest_of_the_record_survives(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    before = load_training_record(model_dir)
    scenario = _scenario(tmp_path)
    alerts = _alerts(tmp_path, ["T1071/T1041"])

    main(["--model-dir", str(model_dir), "--replay", str(scenario), str(alerts)])

    after = load_training_record(model_dir)
    assert after.trained_at == before.trained_at
    assert after.training_script == before.training_script
    assert after.dataset_versions == before.dataset_versions
    assert after.schema_version == before.schema_version


def _env():
    from synthaea_ml.registry.records import Environment

    return Environment(os="linux", kernel="7.2.4", arch="x86_64")
