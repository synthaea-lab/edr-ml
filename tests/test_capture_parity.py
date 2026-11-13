"""Capture-format parity (issue #109), Python side: `aggregate_correlation` must
reproduce `fixtures/capture_correlation_golden.jsonl` from `fixtures/capture_events.jsonl`.

The Rust counterpart is `crates/ml/tests/capture_parity.rs`. If either breaks, the
agent's `events.jsonl` and the trainer disagree on the correlation vector. Regenerate
with `fixtures/generators/gen_capture_correlation_golden.py` only on a deliberate format change.
"""

from pathlib import Path

import pytest
from helpers import read_golden_jsonl

from synthaea_ml.data.aggregate_correlation import charger_events, vecteurs_par_pid
from synthaea_ml.units import NS_PER_SECOND

FIXTURES = Path(__file__).resolve().parent / "fixtures"
S = NS_PER_SECOND


def golden_rows() -> list[dict]:
    return read_golden_jsonl(FIXTURES / "capture_correlation_golden.jsonl")


@pytest.mark.parametrize("row", golden_rows(), ids=lambda r: f"pid{r['pid']}")
def test_vectors_match_golden(row: dict) -> None:
    events = charger_events(FIXTURES / "capture_events.jsonl")
    vectors = {v["pid"]: v["features"] for v in vecteurs_par_pid(events, 60 * S, 1)}
    assert row["pid"] in vectors
    got = vectors[row["pid"]]
    assert len(got) == len(row["features"])
    for g, e in zip(got, row["features"], strict=True):
        assert g == pytest.approx(e, rel=1e-6, abs=1e-9)
