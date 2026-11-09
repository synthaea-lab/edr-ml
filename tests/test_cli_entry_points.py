"""The command-line entry points: every module that runs as `python -m` is a console script,
and every `main` takes its arguments as `argv` and returns the exit status."""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "synthaea_ml"


def _console_scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        scripts: dict[str, str] = tomllib.load(f)["project"]["scripts"]
    return scripts


def _runnable_modules() -> set[str]:
    """The modules of the package with an `if __name__ == "__main__":` block."""
    modules = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
                relative = path.relative_to(ROOT).with_suffix("")
                modules.add(".".join(relative.parts))
    return modules


SCRIPTS = _console_scripts()


def test_every_runnable_module_is_a_console_script() -> None:
    targets = {target.split(":")[0] for target in SCRIPTS.values()}
    assert targets == _runnable_modules()


def test_console_scripts_share_one_prefix() -> None:
    assert all(name.startswith("synthaea-ml-") for name in SCRIPTS)


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_each_console_script_prints_its_help_and_exits_zero(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module_name, function_name = SCRIPTS[name].split(":")
    main = getattr(importlib.import_module(module_name), function_name)

    with pytest.raises(SystemExit) as exited:
        main(["--help"])

    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("usage: ")


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_each_console_script_reports_a_usage_error_with_status_2(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module_name, function_name = SCRIPTS[name].split(":")
    main = getattr(importlib.import_module(module_name), function_name)

    with pytest.raises(SystemExit) as exited:
        main(["--no-such-option"])

    assert exited.value.code == 2
    assert ": error: " in capsys.readouterr().err


# --- the parsed command lines and the exit statuses that used to be `sys.exit` calls ---


def test_the_trainer_options_parse_into_frozen_args(tmp_path: Path) -> None:
    from synthaea_ml.training import train_linux
    from synthaea_ml.training.trainer_args import TrainerArgs

    args = train_linux.parse_args(
        ["--dataset", "a", "b", "--output-dir", str(tmp_path), "--robustness-scenarios", "s.yaml"]
    )

    assert args == TrainerArgs(
        datasets=(Path("a"), Path("b")),
        output_dir=tmp_path,
        robustness_scenarios=(Path("s.yaml"),),
        robustness_events=None,
    )


def test_the_robustness_commands_parse_into_their_own_args(tmp_path: Path) -> None:
    from synthaea_ml.evaluation.robustness_cli import (
        ReportArgs,
        RobustnessThresholds,
        VerifyArgs,
        parse_args,
    )

    verify = parse_args(["verify", "--model-dir", str(tmp_path), "--max-escape-rate", "0.3"])
    report = parse_args(["report", "--model-dir", str(tmp_path)])

    assert verify == VerifyArgs(
        model_dir=tmp_path,
        thresholds=RobustnessThresholds(max_escape_rate=0.3),
        check_regression=False,
    )
    assert report == ReportArgs(model_dir=tmp_path, output=None)


def test_calibrate_llr_returns_1_without_enough_malicious_pids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from synthaea_ml.calibration import calibrate_llr

    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    alerts = tmp_path / "alerts.ndjson"

    status = calibrate_llr.main(["--events", str(events), "--alerts", str(alerts)])

    assert status == 1
    assert "[error] fewer than 1 malicious PID(s)" in capsys.readouterr().err


def test_aggregate_correlation_returns_1_on_a_log_without_usable_events(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from synthaea_ml.data import aggregate_correlation

    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    out = tmp_path / "vectors.jsonl"

    status = aggregate_correlation.main(["--events", str(events), "--out", str(out)])

    assert status == 1
    assert "[error] no usable events in the log." in capsys.readouterr().err
    assert not out.exists()


def test_verify_onnx_returns_1_when_there_is_nothing_to_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from synthaea_ml.export import verify_onnx

    monkeypatch.setattr(verify_onnx, "REGISTRY", tmp_path)

    assert verify_onnx.main([]) == 1
    assert "no model.onnx found under" in capsys.readouterr().err
