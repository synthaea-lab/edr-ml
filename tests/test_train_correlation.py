"""The correlation-vector trainer, end to end on a tiny synthetic dataset.

Synthetic vectors only: they prove the run (dataset -> forest -> ONNX + metadata) holds
together and writes where it is told. They say nothing about detection quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthaea_ml.features.correlation import FEATURE_NAMES
from synthaea_ml.training import train_correlation


def _vectors(n: int = 40) -> list[dict]:
    """`n` distinct quiet-process vectors in the shape `aggregate_correlation` writes."""
    return [
        {
            "pid": 1000 + i,
            "features": [1.0, float(i % 3), 0.0, float(i % 2), float(i % 2), 0.0, 0.5 + i, 1.0 + i % 4],
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

    assert train_correlation.main([]) == 0

    assert model.stat().st_size > 0
    written = json.loads(metadata.read_text(encoding="utf-8"))
    assert written["feature_bounds"]["feature_names"] == list(FEATURE_NAMES)
    assert isinstance(written["threshold"], float)
