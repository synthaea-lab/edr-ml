"""`synthaea-ml-record-replay`: a scenario run's alerts become a replay in the model record
(issue #49), bound to the sidecar the release gate re-verifies."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from helpers import write_baseline_dataset

from synthaea_ml import siblings
from synthaea_ml.evaluation.record_replay import (
    ReplayArgs,
    UnknownScenarioError,
    parse_args,
    record_replay,
    run,
)
from synthaea_ml.registry.layout import MODEL_RECORD_FILENAME
from synthaea_ml.registry.training_record import (
    Environment,
    dataset_version_from_manifest,
    load_training_record,
    verify_scenario_replay,
    write_training_record,
)

_START = datetime(2026, 11, 1, 10, 0, 0, tzinfo=UTC)
_END = datetime(2026, 11, 1, 11, 0, 0, tzinfo=UTC)
_DEBIAN = Environment(os="linux", kernel="6.1.0-26-amd64", arch="x86_64")
_FEDORA = Environment(os="linux", kernel="6.17.1-300.fc43.x86_64", arch="x86_64")

_SIDECAR = """\
name: beacon
platform: linux
script: beacon.sh
expected_detections:
  - technique: "T1071/T1041"
    rule: check_beacon
    min_count: 1
    tolerance: 0
"""


def _model_dir(tmp_path: Path) -> Path:
    dataset = write_baseline_dataset(
        tmp_path / "baselines" / "dev", ['{"argv": ["ls"]}'], capture_start=_START, capture_end=_END
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_training_record(
        model_dir,
        training_script="synthaea_ml/training/train_linux.py",
        dataset_versions=[dataset_version_from_manifest(dataset)],
        trained_at=_START,
    )
    return model_dir


def _sidecars(tmp_path: Path, name: str = "beacon", contents: str = _SIDECAR) -> Path:
    root = tmp_path / "scenarios"
    root.mkdir(exist_ok=True)
    (root / f"{name}.yaml").write_text(contents, encoding="utf-8", newline="\n")
    return root


def _alerts(tmp_path: Path, count: int) -> Path:
    path = tmp_path / f"alerts-{count}.ndjson"
    line = '{"timestamp_ns":1,"technique":"T1071/T1041","message":"m"}\n'
    path.write_text(line * count, encoding="utf-8", newline="\n")
    return path


def _args(tmp_path: Path, model_dir: Path, alerts: Path, **overrides: Any) -> ReplayArgs:
    args = ReplayArgs(
        model_dir=model_dir,
        scenario="beacon",
        alerts=alerts,
        scenarios_root=_sidecars(tmp_path),
        environment=_DEBIAN,
    )
    return replace(args, **overrides)


def test_a_recorded_replay_lands_in_the_model_record(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)

    replay = record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 1)))

    assert replay.passed
    assert load_training_record(model_dir).scenario_replays == [replay]


def test_a_recorded_replay_passes_the_release_gates_scenario_check(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    args = _args(tmp_path, model_dir, _alerts(tmp_path, 1))

    record_replay(args)

    for replay in load_training_record(model_dir).scenario_replays:
        verify_scenario_replay(replay, args.scenarios_root)


def test_a_failing_run_is_recorded_as_failed(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)

    replay = record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 2)))

    assert not replay.passed
    assert load_training_record(model_dir).scenario_replays == [replay]


def test_a_rerun_in_the_same_environment_replaces_the_earlier_replay(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 2)))

    rerun = record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 1)))

    assert load_training_record(model_dir).scenario_replays == [rerun]


def test_a_run_in_another_environment_is_kept_beside_the_first(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    on_debian = record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 1)))

    on_fedora = record_replay(
        _args(tmp_path, model_dir, _alerts(tmp_path, 1), environment=_FEDORA)
    )

    assert load_training_record(model_dir).scenario_replays == [on_debian, on_fedora]


def test_a_scenario_without_a_sidecar_is_refused_and_the_record_is_untouched(
    tmp_path: Path,
) -> None:
    model_dir = _model_dir(tmp_path)
    before = (model_dir / MODEL_RECORD_FILENAME).read_bytes()

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 1), scenario="missing"))

    assert (model_dir / MODEL_RECORD_FILENAME).read_bytes() == before


def test_a_sidecar_whose_name_is_not_its_file_name_is_refused(tmp_path: Path) -> None:
    """The gate looks the sidecar up by the replay's name, so a mismatch could never verify."""
    model_dir = _model_dir(tmp_path)
    root = _sidecars(tmp_path, name="renamed")

    with pytest.raises(UnknownScenarioError, match="beacon"):
        record_replay(
            _args(
                tmp_path,
                model_dir,
                _alerts(tmp_path, 1),
                scenario="renamed",
                scenarios_root=root,
            )
        )


def test_the_record_is_rewritten_with_lf_line_endings(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)

    record_replay(_args(tmp_path, model_dir, _alerts(tmp_path, 1)))

    assert b"\r" not in (model_dir / MODEL_RECORD_FILENAME).read_bytes()


def test_the_sidecars_default_to_the_scenarios_sibling(tmp_path: Path) -> None:
    args = parse_args(
        ["--model-dir", str(tmp_path), "--scenario", "beacon", "--alerts", "a.ndjson"]
    )

    assert args.scenarios_root == siblings.scenario_sidecars()


def test_the_environment_defaults_to_this_machine(tmp_path: Path) -> None:
    args = parse_args(
        ["--model-dir", str(tmp_path), "--scenario", "beacon", "--alerts", "a.ndjson"]
    )

    assert args.environment is None


def test_the_environment_of_a_remote_run_is_taken_from_the_command_line(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--model-dir", str(tmp_path),
            "--scenario", "beacon",
            "--alerts", "a.ndjson",
            "--environment", "linux", "6.1.0-26-amd64", "x86_64",
        ]
    )

    assert args.environment == _DEBIAN


def test_the_command_exits_zero_and_prints_the_outcome_even_when_the_replay_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _model_dir(tmp_path)

    status = run(_args(tmp_path, model_dir, _alerts(tmp_path, 2)))

    assert status == 0
    assert "beacon" in capsys.readouterr().out
