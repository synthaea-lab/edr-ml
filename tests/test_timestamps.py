"""The timestamps a calibration and a robustness card record: their spelling, and (once the
clock is injectable) the value.

The two spellings differ on purpose and are pinned as they are: `calibrated_at` is
`datetime.isoformat()` of a UTC datetime (`+00:00`, microseconds when they are not zero),
`tested_at` and `trained_at` are `%Y-%m-%dT%H:%M:%SZ`. Unifying them changes the bytes of
a model record and needs a schema decision.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from synthaea_ml.calibration import calibrate_threshold
from synthaea_ml.evaluation.robustness import run_robustness_evaluation
from synthaea_ml.timestamps import format_utc

ISO_OFFSET = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?\+00:00")
ISO_Z = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

_SCENARIO = """name: clock_scenario
platform: linux
script: clock.sh
simulates: Scenario for the timestamp tests
expected_detections:
  - technique: "T1071"
    rule: c2
    min_count: 1
    tolerance: 0
"""


class _LinearModel:
    """Scores a row as a fixed linear function of its first and last features."""

    n_features_in_ = 17

    def decision_function(self, rows: list[list[float]]) -> np.ndarray:
        return np.array([0.01 * row[0] - 0.5 + 0.05 * row[-1] for row in rows])


def _calibration_inputs() -> tuple[IsolationForest, np.ndarray]:
    x = np.random.default_rng(42).standard_normal((100, 5)).astype(np.float32)
    return IsolationForest(contamination=0.05, random_state=42).fit(x), x


def _scenario(directory: Path) -> Path:
    path = directory / "clock_scenario.yaml"
    path.write_text(_SCENARIO, encoding="utf-8")
    return path


def test_calibrated_at_defaults_to_now_in_isoformat_with_a_utc_offset() -> None:
    model, x = _calibration_inputs()

    cal = calibrate_threshold(model, x, fp_budget=5.0, benign_rate_per_day=1000.0)

    assert ISO_OFFSET.fullmatch(cal.calibrated_at)


def test_tested_at_defaults_to_now_with_a_z_suffix(tmp_path: Path) -> None:
    card = run_robustness_evaluation(_LinearModel(), _scenario(tmp_path), mutation_seed=7)

    assert ISO_Z.fullmatch(card.tested_at)


def test_calibrated_at_is_the_injected_moment_in_utc() -> None:
    model, x = _calibration_inputs()
    paris = timezone(timedelta(hours=1))

    cal = calibrate_threshold(
        model,
        x,
        fp_budget=5.0,
        benign_rate_per_day=1000.0,
        calibrated_at=datetime(2026, 9, 11, 13, 0, 0, tzinfo=paris),
    )

    assert cal.calibrated_at == "2026-09-11T12:00:00+00:00"


def test_calibrated_at_keeps_the_microseconds_of_the_injected_moment() -> None:
    model, x = _calibration_inputs()

    cal = calibrate_threshold(
        model,
        x,
        fp_budget=5.0,
        benign_rate_per_day=1000.0,
        calibrated_at=datetime(2026, 9, 11, 12, 0, 0, 250_000, tzinfo=UTC),
    )

    assert cal.calibrated_at == "2026-09-11T12:00:00.250000+00:00"


def test_a_naive_calibrated_at_is_refused() -> None:
    model, x = _calibration_inputs()

    with pytest.raises(ValueError, match="calibrated_at must be timezone-aware"):
        calibrate_threshold(
            model,
            x,
            fp_budget=5.0,
            benign_rate_per_day=1000.0,
            calibrated_at=datetime(2026, 9, 11, 12, 0, 0),  # noqa: DTZ001 - naive on purpose  # noqa: DTZ001 - naive on purpose
        )


def test_tested_at_is_the_injected_moment_to_the_second_in_utc(tmp_path: Path) -> None:
    paris = timezone(timedelta(hours=1))

    card = run_robustness_evaluation(
        _LinearModel(),
        _scenario(tmp_path),
        mutation_seed=7,
        tested_at=datetime(2026, 9, 11, 13, 0, 0, 999_999, tzinfo=paris),
    )

    assert card.tested_at == "2026-09-11T12:00:00Z"


def test_a_naive_tested_at_is_refused_before_the_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        run_robustness_evaluation(
            _LinearModel(),
            tmp_path / "missing.yaml",
            tested_at=datetime(2026, 9, 11, 12, 0, 0),  # noqa: DTZ001 - naive on purpose
        )


def test_format_utc_spells_utc_with_a_z_suffix() -> None:
    assert format_utc(datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)) == "2026-09-11T12:00:00Z"
    assert (
        format_utc(datetime(2026, 9, 11, 0, 30, tzinfo=timezone(timedelta(hours=2))))
        == "2026-09-10T22:30:00Z"
    )


def test_format_utc_refuses_a_naive_moment() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        format_utc(datetime(2026, 9, 11, 12, 0, 0))  # noqa: DTZ001 - naive on purpose
