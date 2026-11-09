"""Golden parity test: `extract_behavior_features` must reproduce `behavior_golden.jsonl`.

The file pins the behavior vector of `BehaviorVector::from_window` (Rust, correlator) on the
cases both sides agree on. See `tests/fixtures/generators/gen_behavior_golden.py` for what is
left out and what to do when this fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import read_golden_jsonl

from synthaea_ml.features.behavior import extract_behavior_features

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "behavior_golden.jsonl"
# The order of `BehaviorVector::to_vec()`.
NAMES = (
    "has_exec",
    "has_connect",
    "has_filewrite",
    "time_exec_to_connect_ms",
    "time_exec_to_filewrite_ms",
    "is_suspicious_path",
    "connect_count",
    "distinct_dports",
    "dest_is_external",
)


def golden_rows() -> list[dict]:
    return read_golden_jsonl(GOLDEN)


@pytest.mark.parametrize("row", golden_rows(), ids=lambda r: r["name"])
def test_behavior_features_match_golden(row: dict) -> None:
    got = extract_behavior_features(row["events"])

    assert len(got) == len(row["features"]) == len(NAMES)
    for name, g, e in zip(NAMES, got, row["features"], strict=True):
        assert g == pytest.approx(e, rel=1e-6, abs=1e-9), (
            f"feature {name!r} drifted for case {row['name']!r}: {g} != {e}"
        )


def test_golden_case_names_are_unique() -> None:
    names = [r["name"] for r in golden_rows()]
    assert len(names) == len(set(names))


def test_golden_covers_every_feature() -> None:
    """Every feature is non-zero in at least one case, or the file does not lock that column."""
    rows = golden_rows()
    for i, name in enumerate(NAMES):
        assert any(r["features"][i] != 0.0 for r in rows), f"no golden case exercises {name!r}"
