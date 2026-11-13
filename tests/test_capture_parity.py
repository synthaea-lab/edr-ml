"""Capture-format parity (issue #109), Python side: `aggregate_correlation` must
reproduce `fixtures/capture_correlation_golden.jsonl` from `fixtures/capture_events.jsonl`.

The Rust counterpart is `crates/ml/tests/capture_parity.rs`. If either breaks, the
agent's `events.jsonl` and the trainer disagree on the correlation vector. Regenerate
with `fixtures/generators/gen_capture_correlation_golden.py` only on a deliberate format change.
"""

from pathlib import Path

import pytest
from helpers import read_golden_jsonl

from synthaea_ml.data.aggregate_correlation import load_events, vectors_by_pid

FIXTURES = Path(__file__).resolve().parent / "fixtures"
S = 1_000_000_000


def golden_rows() -> list[dict]:
    return read_golden_jsonl(FIXTURES / "capture_correlation_golden.jsonl")


@pytest.mark.parametrize("row", golden_rows(), ids=lambda r: f"pid{r['pid']}")
def test_vectors_match_golden(row: dict) -> None:
    events = load_events(FIXTURES / "capture_events.jsonl")
    vectors = {v["pid"]: v["features"] for v in vectors_by_pid(events, 60 * S, 1)}
    assert row["pid"] in vectors
    got = vectors[row["pid"]]
    assert len(got) == len(row["features"])
    for g, e in zip(got, row["features"], strict=True):
        assert g == pytest.approx(e, rel=1e-6, abs=1e-9)


def test_deprecated_french_aliases_point_at_the_english_names() -> None:
    # The old French names stay importable for one release (#38). Remove with the alias.
    from synthaea_ml.data import aggregate_correlation as ac

    assert ac.charger_events is ac.load_events
    assert ac.vecteurs_par_pid is ac.vectors_by_pid
