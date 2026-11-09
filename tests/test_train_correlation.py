"""The correlation-vector trainer, end to end on a tiny synthetic dataset.

Synthetic vectors only: they prove the run (dataset -> forest -> ONNX + metadata) holds
together and writes where it is told. They say nothing about detection quality.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from helpers import write_baseline_dataset

from synthaea_ml.data import aggregate_correlation
from synthaea_ml.data.manifest import CORRELATION_DATASET_FILENAME, hash_hostname
from synthaea_ml.features.correlation import FEATURE_NAMES
from synthaea_ml.registry.layout import METADATA_FILENAME, MODEL_FILENAME
from synthaea_ml.registry.training_record import load_training_record, verify_training_record
from synthaea_ml.training import train_correlation
from synthaea_ml.training.config import CORRELATION_FOREST

_START = datetime(2026, 11, 2, 10, 0, 0, tzinfo=UTC)
_END = datetime(2026, 11, 2, 11, 0, 0, tzinfo=UTC)
_DATASET_NAME = f"linux__dev__{hash_hostname('testhost')}__2026-11-02"
"""The name the manifest derives, so the provenance gate finds the directory by it."""


def _vectors(n: int = 40) -> list[dict]:
    """`n` distinct quiet-process vectors in the shape `aggregate_correlation` writes."""
    return [
        {
            "pid": 1000 + i,
            "features": [1.0, i % 3, 0.0, i % 2, i % 2, 0.0, 0.5 + i, 1.0 + i % 4],
            "feature_names": list(FEATURE_NAMES),
        }
        for i in range(n)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_the_legacy_run_reads_and_writes_its_module_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _write_jsonl(tmp_path / "data" / "correlation_train.jsonl", _vectors())
    model = tmp_path / "model_correlation.onnx"
    metadata = tmp_path / "model_correlation_metadata.json"
    monkeypatch.setattr(train_correlation, "DATA_PATH", data)
    monkeypatch.setattr(train_correlation, "MODEL_PATH", model)
    monkeypatch.setattr(train_correlation, "METADATA_PATH", metadata)

    with pytest.warns(FutureWarning, match="--dataset"):
        assert train_correlation.main([]) == 0

    assert model.stat().st_size > 0
    written = json.loads(metadata.read_text(encoding="utf-8"))
    assert written["feature_bounds"]["feature_names"] == list(FEATURE_NAMES)
    assert isinstance(written["threshold"], float)


def _write_dataset(root: Path, rows: list[dict]) -> Path:
    return write_baseline_dataset(
        root / _DATASET_NAME,
        [json.dumps(r) for r in rows],
        capture_start=_START,
        capture_end=_END,
        baseline_filename=CORRELATION_DATASET_FILENAME,
    )


def test_a_dataset_run_writes_the_model_metadata_and_record_into_the_output_dir(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    dataset = _write_dataset(tmp_path, _vectors())
    out = tmp_path / "registry" / "correlation-iforest-linux" / "0.1.0"

    assert train_correlation.main(["--dataset", str(dataset), "--output-dir", str(out)]) == 0

    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]
    assert (out / MODEL_FILENAME).stat().st_size > 0
    metadata = json.loads((out / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["feature_bounds"]["feature_names"] == list(FEATURE_NAMES)
    record = load_training_record(out)
    assert record.training_script == "synthaea_ml/training/train_correlation.py"
    assert record.hyperparameters == CORRELATION_FOREST.hyperparameters()
    assert [d.name for d in record.dataset_versions] == [_DATASET_NAME]
    assert record.dataset_versions[0].sample_count == 40
    assert record.conformal_calibration is not None
    assert record.conformal_calibration.threshold == metadata["threshold"]
    verify_training_record(out, baselines_root=tmp_path, scenarios_root=tmp_path)


def test_a_dataset_edited_after_its_manifest_is_rejected_before_any_model(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path, _vectors())
    with (dataset / CORRELATION_DATASET_FILENAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(_vectors(41)[40]) + "\n")
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="hash mismatch"):
        train_correlation.main(["--dataset", str(dataset), "--output-dir", str(out)])
    assert not (out / MODEL_FILENAME).exists()


@pytest.mark.parametrize("argv", [["--dataset", "d"], ["--output-dir", "o"]])
def test_dataset_and_output_dir_go_together(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        train_correlation.parse_args(argv)
    assert exit_info.value.code == 2


def test_the_aggregator_default_is_where_the_legacy_trainer_reads(tmp_path: Path) -> None:
    assert train_correlation.DATA_PATH == aggregate_correlation.LEGACY_OUT_PATH
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")

    with pytest.warns(FutureWarning, match="--out"):
        args = aggregate_correlation.parse_args(["--events", str(events)])

    assert args.out == train_correlation.DATA_PATH
