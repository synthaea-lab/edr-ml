"""Lower layers must not pull in the training entry points.

Pins the moves of the shared sanity samples, the behavior minimum sample count and the
common-set timestamp helper out of the trainer scripts.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from synthaea_ml.data import common_set
from synthaea_ml.features.sanity_samples import LINUX_SANITY_SAMPLES, WINDOWS_SANITY_SAMPLES
from synthaea_ml.training import config, train_linux, train_windows


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("synthaea_ml.export.verify_onnx", "synthaea_ml.training"),
        ("synthaea_ml.evaluation.t1_lineage_gain", "synthaea_ml.training.train_"),
    ],
)
def test_module_does_not_import_training_scripts(module: str, forbidden: str) -> None:
    code = (
        f"import sys, {module}\n"
        f"leaked = sorted(m for m in sys.modules if m.startswith({forbidden!r}))\n"
        "print(','.join(leaked))\n"
    )
    out = subprocess.run(  # noqa: S603 - fixed interpreter, code built from test literals
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"{module} imports {out}"


def test_trainers_print_the_shared_sanity_samples() -> None:
    assert train_linux.SANITY_CHECK_SAMPLES is LINUX_SANITY_SAMPLES
    assert train_windows.SANITY_CHECK_SAMPLES is WINDOWS_SANITY_SAMPLES


def test_behavior_min_samples_is_unchanged() -> None:
    assert config.BEHAVIOR_MIN_SAMPLES == 20


def test_timestamp_ns_is_public_and_integer_exact() -> None:
    assert common_set.timestamp_ns({"timestamp": "2026-08-24T00:00:00.123456Z"}) == (
        1_787_529_600_123_456_000
    )
    assert common_set.timestamp_ns({"timestamp": "1970-01-01T00:00:01"}) == 1_000_000_000
