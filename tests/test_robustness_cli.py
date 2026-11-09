"""Tests for robustness CLI (robustness_cli.py)."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from synthaea_ml.evaluation.robustness_cli import (
    Gate,
    GateFailure,
    RobustnessThresholds,
    evaluate_robustness_gates,
    main,
    previous_version,
    render_verification,
    verify_robustness,
)
from synthaea_ml.registry.training_record import (
    DatasetVersion,
    MutationTestResult,
    RobustnessCard,
    TrainingRecord,
    record_to_dict,
)


def _create_test_record_with_robustness(
    scenario_name: str = "test_scenario",
    escape_rate: float = 0.10,
    median_score_degradation: float = 0.10,
) -> TrainingRecord:
    """Helper to create a TrainingRecord with robustness card."""
    mutation_result = MutationTestResult(
        mutation_class="base64_encode",
        intensity="light",
        original_score=-0.1,
        mutated_score=0.05,
        score_delta=0.15,
        threshold=0.0,
        escaped=True,
        seed=42,
    )

    robustness_card = RobustnessCard(
        scenario_name=scenario_name,
        scenario_yaml_sha256="abc" * 21 + "d",
        tested_at="2026-09-23T10:00:00Z",
        mutation_results=[mutation_result],
        escape_rate=escape_rate,
        median_score_degradation=median_score_degradation,
        worst_case_degradation=0.15,
    )

    dataset_version = DatasetVersion(
        name="test_dataset",
        baseline_sha256="def" * 21 + "e",
        sample_count=100,
    )

    return TrainingRecord(
        schema_version=3,
        trained_at="2026-09-23T10:00:00Z",
        training_script="test.py",
        dataset_versions=[dataset_version],
        robustness_cards=[robustness_card],
    )


def testprevious_version(tmp_path: Path) -> None:
    """Test finding previous version directory."""
    # Create version directories
    registry_dir = tmp_path / "cmdline-iforest-linux"
    registry_dir.mkdir()

    (registry_dir / "0.1.0").mkdir()
    (registry_dir / "0.2.0").mkdir()
    (registry_dir / "0.2.1").mkdir()
    (registry_dir / "0.3.0").mkdir()

    # Test finding previous patch version
    prev = previous_version(registry_dir / "0.2.1")
    assert prev == registry_dir / "0.2.0"

    # Test finding previous minor version
    prev = previous_version(registry_dir / "0.3.0")
    # Should find 0.2.* (latest patch in previous minor)
    assert prev is not None
    assert prev.name.startswith("0.2")

    # Test no previous version
    prev = previous_version(registry_dir / "0.1.0")
    assert prev is None


def test_verify_robustness_pass(tmp_path: Path) -> None:
    """Test verify_robustness passes with good metrics."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    record = _create_test_record_with_robustness(escape_rate=0.08)
    record_path = model_dir / "model_record.json"
    record_path.write_text(json.dumps(record_to_dict(record)), encoding="utf-8")

    result = verify_robustness(model_dir, RobustnessThresholds(0.15, 0.20))

    assert result.passed
    assert result.failures == []
    report = render_verification(result)
    assert "PASS" in report
    assert "8.00%" in report  # escape rate


def test_verify_robustness_fail_escape_rate(tmp_path: Path) -> None:
    """Test verify_robustness fails when escape rate exceeds threshold."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    record = _create_test_record_with_robustness(escape_rate=0.20)
    record_path = model_dir / "model_record.json"
    record_path.write_text(json.dumps(record_to_dict(record)), encoding="utf-8")

    result = verify_robustness(model_dir, RobustnessThresholds(0.15, 0.20))

    assert not result.passed
    assert result.failures == [GateFailure("test_scenario", Gate.ESCAPE_RATE, 0.20, 0.15)]
    report = render_verification(result)
    assert "FAIL" in report
    assert "20.00%" in report  # escape rate


def test_verify_robustness_regression(tmp_path: Path) -> None:
    """Test regression checking between versions."""
    registry_dir = tmp_path / "cmdline-iforest-linux"
    registry_dir.mkdir()

    # Create previous version (0.1.0) with 8% escape rate
    prev_dir = registry_dir / "0.1.0"
    prev_dir.mkdir()
    prev_record = _create_test_record_with_robustness(escape_rate=0.08)
    (prev_dir / "model_record.json").write_text(
        json.dumps(record_to_dict(prev_record)), encoding="utf-8"
    )

    # Create current version (0.2.0) with 10% escape rate (2% increase, within threshold)
    curr_dir = registry_dir / "0.2.0"
    curr_dir.mkdir()
    curr_record = _create_test_record_with_robustness(escape_rate=0.10)
    (curr_dir / "model_record.json").write_text(
        json.dumps(record_to_dict(curr_record)), encoding="utf-8"
    )

    # Should pass (2% increase < 5% threshold)
    result = verify_robustness(
        curr_dir, RobustnessThresholds(max_regression_increase=0.05), check_regression=True
    )
    assert result.passed
    report = render_verification(result)
    assert "Regression check" in report
    assert "8.00% → 10.00%" in report


def test_verify_robustness_regression_fail(tmp_path: Path) -> None:
    """Test regression check fails when escape rate increases too much."""
    registry_dir = tmp_path / "cmdline-iforest-linux"
    registry_dir.mkdir()

    # Create previous version with 8% escape rate
    prev_dir = registry_dir / "0.1.0"
    prev_dir.mkdir()
    prev_record = _create_test_record_with_robustness(escape_rate=0.08)
    (prev_dir / "model_record.json").write_text(
        json.dumps(record_to_dict(prev_record)), encoding="utf-8"
    )

    # Create current version with 20% escape rate (12% increase, exceeds threshold)
    curr_dir = registry_dir / "0.2.0"
    curr_dir.mkdir()
    curr_record = _create_test_record_with_robustness(escape_rate=0.20)
    (curr_dir / "model_record.json").write_text(
        json.dumps(record_to_dict(curr_record)), encoding="utf-8"
    )

    # Should fail (12% increase > 5% threshold)
    result = verify_robustness(
        curr_dir, RobustnessThresholds(max_regression_increase=0.05), check_regression=True
    )
    assert not result.passed
    assert [failure.gate for failure in result.failures] == [Gate.ESCAPE_RATE, Gate.REGRESSION]
    assert "REGRESSION" in render_verification(result)


# --- Pinned behavior -------------------------------------------------------------
# The tests below pin the exact `verify` output and the previous-version lookup, so the
# command/query split of the verifier can be checked byte for byte.


def _write_record(model_dir: Path, record: TrainingRecord) -> Path:
    model_dir.mkdir(parents=True)
    (model_dir / "model_record.json").write_text(
        json.dumps(record_to_dict(record)), encoding="utf-8"
    )
    return model_dir


def _with_cards(*records: TrainingRecord) -> TrainingRecord:
    """The first record, carrying the robustness cards of all of them in order."""
    cards = [card for record in records for card in record.robustness_cards]
    return dataclasses.replace(records[0], robustness_cards=cards)


def _run_verify(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["robustness_cli", "verify", *args])
    with pytest.raises(SystemExit) as exit_info:
        main()
    return int(exit_info.value.code or 0), capsys.readouterr().out


def test_verify_cli_prints_metrics_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _write_record(
        tmp_path / "model", _create_test_record_with_robustness(escape_rate=0.08)
    )

    exit_code, out = _run_verify(monkeypatch, capsys, "--model-dir", str(model_dir))

    assert exit_code == 0
    assert out == (
        "Verifying robustness for 1 scenario(s)...\n"
        "\n"
        "test_scenario:\n"
        "  Escape rate: 8.00%\n"
        "  Median degradation: +0.100\n"
        "  Worst case degradation: +0.150\n"
        "\n"
        "PASS: All robustness checks passed\n"
    )


def test_verify_cli_reports_every_threshold_failure_per_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _with_cards(
        _create_test_record_with_robustness("both", 0.20, median_score_degradation=0.30),
        _create_test_record_with_robustness("fine", 0.01),
    )
    model_dir = _write_record(tmp_path / "model", record)

    exit_code, out = _run_verify(monkeypatch, capsys, "--model-dir", str(model_dir))

    assert exit_code == 1
    assert out == (
        "Verifying robustness for 2 scenario(s)...\n"
        "\n"
        "both:\n"
        "  Escape rate: 20.00%\n"
        "  Median degradation: +0.300\n"
        "  Worst case degradation: +0.150\n"
        "  FAIL: escape rate 20.00% > 15.00%\n"
        "  FAIL: median degradation +0.300 > 0.200\n"
        "\n"
        "fine:\n"
        "  Escape rate: 1.00%\n"
        "  Median degradation: +0.100\n"
        "  Worst case degradation: +0.150\n"
        "\n"
        "FAIL: Robustness checks failed\n"
    )


def test_verify_cli_fails_a_record_without_robustness_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = dataclasses.replace(_create_test_record_with_robustness(), robustness_cards=[])
    model_dir = _write_record(tmp_path / "model", record)

    exit_code, out = _run_verify(
        monkeypatch, capsys, "--model-dir", str(model_dir), "--no-regression"
    )

    assert exit_code == 1
    assert out == f"ERROR: No robustness cards found in {model_dir}\n"


def test_verify_cli_regression_compares_scenarios_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_dir = tmp_path / "cmdline-iforest-linux"
    _write_record(
        registry_dir / "0.1.0",
        _with_cards(
            _create_test_record_with_robustness("steady", 0.08),
            _create_test_record_with_robustness("worse", 0.02),
            _create_test_record_with_robustness("retired", 0.50),
        ),
    )
    model_dir = _write_record(
        registry_dir / "0.2.0",
        _with_cards(
            _create_test_record_with_robustness("steady", 0.10),
            _create_test_record_with_robustness("worse", 0.12),
            _create_test_record_with_robustness("added", 0.03),
        ),
    )

    exit_code, out = _run_verify(
        monkeypatch,
        capsys,
        "--model-dir",
        str(model_dir),
        "--no-regression",
        "--max-regression-increase",
        "0.05",
    )

    assert exit_code == 1
    assert out.endswith(
        "\n"
        "Regression check against previous version: 0.1.0\n"
        "  steady: 8.00% → 10.00% (Δ +2.00%)\n"
        "  worse: 2.00% → 12.00% (Δ +10.00%)\n"
        "    REGRESSION: escape rate increased by 10.00% (max allowed: 5.00%)\n"
        "  added: no previous baseline (new scenario)\n"
        "\n"
        "FAIL: Robustness checks failed\n"
    )


def test_verify_cli_regression_warns_without_a_previous_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _write_record(
        tmp_path / "registry" / "0.1.0", _create_test_record_with_robustness(escape_rate=0.08)
    )

    exit_code, out = _run_verify(
        monkeypatch, capsys, "--model-dir", str(model_dir), "--no-regression"
    )

    assert exit_code == 0
    assert out.endswith(
        "  Worst case degradation: +0.150\n"
        "\n"
        "WARNING: No previous version found for regression check\n"
        "\n"
        "PASS: All robustness checks passed\n"
    )


def test_verify_cli_regression_warns_when_the_previous_record_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_dir = tmp_path / "registry"
    (registry_dir / "0.1.0").mkdir(parents=True)
    model_dir = _write_record(
        registry_dir / "0.1.1", _create_test_record_with_robustness(escape_rate=0.08)
    )

    exit_code, out = _run_verify(
        monkeypatch, capsys, "--model-dir", str(model_dir), "--no-regression"
    )

    assert exit_code == 0
    assert out.endswith(
        "\n"
        "WARNING: Could not load previous version for regression check: "
        f"model record not found: {registry_dir / '0.1.0' / 'model_record.json'}\n"
        "\n"
        "PASS: All robustness checks passed\n"
    )


def _make_version_dirs(registry_dir: Path, *versions: str) -> None:
    for version in versions:
        (registry_dir / version).mkdir(parents=True)


def test_previous_version_prefers_the_previous_patch(tmp_path: Path) -> None:
    _make_version_dirs(tmp_path, "0.2.5", "0.3.0", "0.3.1")

    assert previous_version(tmp_path / "0.3.1") == tmp_path / "0.3.0"


def test_previous_version_skips_a_missing_patch_to_the_previous_minor(tmp_path: Path) -> None:
    # Only patch - 1 is tried in the current minor: 0.3.0 is not found from 0.3.2.
    _make_version_dirs(tmp_path, "0.3.0", "0.3.2")

    assert previous_version(tmp_path / "0.3.2") is None


def test_previous_version_picks_one_of_several_previous_minor_patches(tmp_path: Path) -> None:
    # Pins current behavior: the first directory glob yields, which depends on the
    # filesystem, not the highest patch. A follow-up makes this pick 0.2.1.
    _make_version_dirs(tmp_path, "0.2.0", "0.2.1", "0.3.0")

    assert previous_version(tmp_path / "0.3.0") in {tmp_path / "0.2.0", tmp_path / "0.2.1"}


def test_previous_version_falls_back_to_the_previous_major(tmp_path: Path) -> None:
    _make_version_dirs(tmp_path, "0.9.0", "1.0.0")

    assert previous_version(tmp_path / "1.0.0") == tmp_path / "0.9.0"


def test_previous_version_ignores_non_semver_directories(tmp_path: Path) -> None:
    _make_version_dirs(tmp_path, "0.1.0", "latest", "0.2")

    assert previous_version(tmp_path / "latest") is None
    assert previous_version(tmp_path / "0.2") is None


def test_check_regression_is_the_new_name_of_no_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = _write_record(
        tmp_path / "registry" / "0.1.0", _create_test_record_with_robustness(escape_rate=0.08)
    )

    _, old_name = _run_verify(monkeypatch, capsys, "--model-dir", str(model_dir), "--no-regression")
    _, new_name = _run_verify(
        monkeypatch, capsys, "--model-dir", str(model_dir), "--check-regression"
    )

    assert new_name == old_name
    assert "regression check" in new_name


def test_evaluate_robustness_gates_lists_thresholds_then_regressions() -> None:
    previous = _with_cards(
        _create_test_record_with_robustness("a", 0.01),
        _create_test_record_with_robustness("b", 0.01),
    )
    record = _with_cards(
        _create_test_record_with_robustness("a", 0.20),
        _create_test_record_with_robustness("b", 0.02, median_score_degradation=0.25),
        _create_test_record_with_robustness("c", 0.50),
    )

    failures = evaluate_robustness_gates(record, previous, RobustnessThresholds())

    assert [(f.scenario_name, f.gate) for f in failures] == [
        ("a", Gate.ESCAPE_RATE),
        ("b", Gate.MEDIAN_DEGRADATION),
        ("c", Gate.ESCAPE_RATE),
        ("a", Gate.REGRESSION),
    ]
    assert failures[-1].observed == pytest.approx(0.19)
    assert failures[-1].limit == 0.05


def test_evaluate_robustness_gates_skips_regressions_without_a_previous_record() -> None:
    record = _create_test_record_with_robustness(escape_rate=0.10)

    assert evaluate_robustness_gates(record, None, RobustnessThresholds()) == []
