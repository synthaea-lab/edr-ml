"""The per-site recalibration entry point (#49) runs end to end.

`train_site_model.main` passed a manifest *file* to `dataset_version_from_manifest` (which
takes the baseline directory) and called `write_training_record(output_dir=...)` (the
parameter is `model_dir`), so every real run died with an exception after training and
exporting. Nothing exercised it, so the loop #49 was closed on could not complete.

The global-model floor is exercised with a stubbed robustness run: the site model (the
`IsolationForest` just trained) and the global model (whatever `--global-model` holds) each
get a canned robustness card.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import robustness_card
from sklearn.ensemble import IsolationForest

from synthaea_ml.data.manifest import write_manifest
from synthaea_ml.registry.training_record import load_training_record
from synthaea_ml.training import train_site_model
from synthaea_ml.training.config import FLOOR_TOLERANCE

_GLOBAL_SAMPLES = [
    {"argv": ["whoami"]},
    {"argv": ["id"]},
    {"argv": ["ls", "-la"]},
    {"argv": ["bash", "/tmp/x.sh"]},
    {"argv": ["cat", "/etc/hosts"]},
    {"argv": ["ps", "-ef"]},
    {"argv": ["uname", "-a"]},
    {"argv": ["/usr/bin/env", "python3"]},
    {"argv": ["curl", "-fSL", "http://x/y"]},
    {"argv": ["grep", "-r", "foo", "/tmp"]},
]


def _write_global_baseline(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "baseline.jsonl").write_text(
        "\n".join(json.dumps(s) for s in _GLOBAL_SAMPLES) + "\n", encoding="utf-8"
    )
    write_manifest(
        directory,
        platform="linux",
        os_version="test",
        workload_label="dev",
        capture_start=datetime(2026, 9, 11, 10, tzinfo=UTC),
        capture_end=datetime(2026, 9, 11, 11, tzinfo=UTC),
        hostname="testhost",
    )


def _write_site_corpus(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": "2026-09-23T10:00:00Z",
                    "event_type": "exec",
                    "agent_id": "a1",
                    "event": {"argv": ["/opt/site/tool", f"--job{i}"]},
                }
            )
            for i in range(8)
        )
        + "\n",
        encoding="utf-8",
    )


def _run_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra_args: str
) -> Path:
    """Run `train_site_model.main()` on a small global baseline and site corpus; returns the
    output directory. `extra_args` are appended after the required arguments."""
    global_dir = tmp_path / "linux__dev__abc__2026-09-11"
    _write_global_baseline(global_dir)
    corpus = tmp_path / "corpus-v1.jsonl"
    _write_site_corpus(corpus)
    out = tmp_path / "site-model"
    monkeypatch.setattr(
        "sys.argv",
        [
            train_site_model.TRAINING_SCRIPT,
            "--global-dataset",
            str(global_dir),
            "--site-corpus",
            str(corpus),
            "--output-dir",
            str(out),
            "--site-name",
            "acme",
            *extra_args,
        ],
    )
    train_site_model.main()
    return out


def _stub_robustness(
    monkeypatch: pytest.MonkeyPatch, site: SimpleNamespace, global_: SimpleNamespace
) -> list[object]:
    """Give the site model (the trained forest) `site` and any other model `global_` as their
    robustness cards. Returns the models evaluated, in call order."""
    evaluated: list[object] = []

    def fake(model, scenario_yaml, tier="T0", mutation_seed=42, threshold=None, events_source=None):
        evaluated.append(model)
        return site if isinstance(model, IsolationForest) else global_

    monkeypatch.setattr(train_site_model, "run_robustness_evaluation", fake)
    return evaluated


def _floor_args(tmp_path: Path) -> list[str]:
    """`--robustness-scenarios` and `--global-model` pointing at a scenario and a pickled
    stand-in for the global model."""
    scenario = tmp_path / "beacon.yaml"
    scenario.write_text("name: beacon\n", encoding="utf-8")
    global_model = tmp_path / "global.pkl"
    global_model.write_bytes(pickle.dumps({"stand-in": "global model"}))
    return ["--robustness-scenarios", str(scenario), "--global-model", str(global_model)]


def test_the_site_trainer_completes_and_records_both_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _run_trainer(monkeypatch, tmp_path)

    assert (out / "model.onnx").exists()
    record = load_training_record(out)
    assert [d.name for d in record.dataset_versions][-1] == "site-corpus-acme"
    assert len(record.dataset_versions) == 2


def test_a_site_model_within_tolerance_of_the_global_model_passes_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluated = _stub_robustness(
        monkeypatch,
        site=robustness_card(0.10 + FLOOR_TOLERANCE.escape_rate, 0.20 + FLOOR_TOLERANCE.median_degradation),
        global_=robustness_card(0.10, 0.20),
    )

    out = _run_trainer(monkeypatch, tmp_path, *_floor_args(tmp_path))

    assert (out / "model.onnx").exists()
    assert [isinstance(m, IsolationForest) for m in evaluated] == [False, True]
    assert evaluated[0] == {"stand-in": "global model"}
    assert "Global-model-floor check PASSED" in capsys.readouterr().out


def test_a_site_model_that_escapes_more_than_the_global_model_fails_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_robustness(monkeypatch, site=robustness_card(0.30, 0.20), global_=robustness_card(0.10, 0.20))

    with pytest.raises(ValueError, match="Global-model-floor check failed"):
        _run_trainer(monkeypatch, tmp_path, *_floor_args(tmp_path))

    out = capsys.readouterr().out
    assert "Global-model-floor check FAILED" in out
    assert "Escape rate regression: 30.00% > 10.00% + 5%" in out
    assert "Median degradation regression" not in out
    assert not (tmp_path / "site-model" / "model.onnx").exists()


def test_a_site_model_that_degrades_more_than_the_global_model_fails_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_robustness(monkeypatch, site=robustness_card(0.10, 0.50), global_=robustness_card(0.10, 0.20))

    with pytest.raises(ValueError, match="Global-model-floor check failed"):
        _run_trainer(monkeypatch, tmp_path, *_floor_args(tmp_path))

    out = capsys.readouterr().out
    assert "Median degradation regression: 0.500 > 0.200 + 0.10" in out
    assert "Escape rate regression" not in out


@pytest.mark.parametrize("given", ["--robustness-scenarios", "--global-model"])
def test_the_floor_is_skipped_with_a_warning_when_only_one_of_its_inputs_is_given(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    given: str,
) -> None:
    evaluated = _stub_robustness(monkeypatch, site=robustness_card(1.0, 1.0), global_=robustness_card(0.0, 0.0))
    floor_args = _floor_args(tmp_path)
    one_input = floor_args[floor_args.index(given) : floor_args.index(given) + 2]

    out = _run_trainer(monkeypatch, tmp_path, *one_input)

    assert (out / "model.onnx").exists()
    assert evaluated == []
    assert (
        "Skipping global-model-floor check (need both --robustness-scenarios and --global-model)"
        in capsys.readouterr().out
    )
