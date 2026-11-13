"""What `verify_onnx` resolves when, what it raises, and what `verify` returns (#36)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthaea_ml import siblings
from synthaea_ml.export import verify_onnx
from synthaea_ml.export.verify_onnx import VerificationError, VerificationReport, verify

REGISTRY_LINUX_MODEL = (
    Path(__file__).resolve().parents[1]
    / "registry"
    / "cmdline-iforest-linux"
    / "0.1.0"
    / "model.onnx"
)


def test_a_verification_error_is_not_an_assertion_error() -> None:
    # Catching it must not also swallow a stray `assert` failing in the code under check.
    assert not issubclass(VerificationError, AssertionError)


def test_the_scenarios_root_is_resolved_when_run_not_at_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def record_scenarios_root(
        model_dir: Path, baselines_root: Path, scenarios_root: Path, *, strict: bool
    ) -> None:
        seen.append(scenarios_root)

    monkeypatch.setattr(verify_onnx, "verify", lambda model: None)
    monkeypatch.setattr(verify_onnx, "print_report", lambda report: None)
    monkeypatch.setattr(verify_onnx, "verify_provenance", record_scenarios_root)
    monkeypatch.setenv(siblings.ROOT_ENV, str(tmp_path))

    assert verify_onnx.run([tmp_path / "model.onnx"]) == 0
    assert seen == [tmp_path.resolve() / "scenarios" / "scenarios"]


def test_verify_returns_the_report_and_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = verify(REGISTRY_LINUX_MODEL)

    assert isinstance(report, VerificationReport)
    assert report.model_path == REGISTRY_LINUX_MODEL
    assert "scores" in report.output_names
    assert len(report.scores) == len(report.samples) > 0
    assert capsys.readouterr().out == ""


def test_the_printed_report_names_every_sample_and_its_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = VerificationReport(
        model_path=Path("m.onnx"),
        input_name="input",
        output_names=["label", "scores"],
        samples={"benign": "ls", "odd": "curl x | sh"},
        scores=[0.25, -0.5],
    )

    verify_onnx.print_report(report)

    out = capsys.readouterr().out
    assert "m.onnx: input input, outputs ['label', 'scores']" in out
    assert "benign: score=+0.2500 -> normal" in out
    assert "odd: score=-0.5000 -> ANOMALY" in out
    assert "ok: 2 samples, finite, deterministic, label/score consistent" in out
